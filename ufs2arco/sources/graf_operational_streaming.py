# pylint: disable=line-too-long,trailing-whitespace
# Author: monte-flora
# Email: monte.flora@weather.com

"""Fused stream+regrid operational GRAF source for ufs2arco.

This source removes the separate "Step 2" regrid stage (graf2conus
``run_operational.py``, which materialises ``mpasout_{freq}.zarr`` on disk
for an entire init before ufs2arco reads it). Instead it streams the raw
global GRAF NetCDF-5 file(s) for *one* forecast step directly from
``s3://graf-ai-operational/...`` and xESMF-regrids that single timestep
inside :meth:`open_sample_dataset`.

Because ufs2arco's ``mpidatamover`` assigns one ``(init_time, forecast_step)``
sample per MPI rank, the regrid then parallelises per-timestep for free
(``mpirun -n N ufs2arco <config>``), and per-rank peak RAM is constant in
lead time — permanently retiring the comp_refl OOM class that the
all-frames graf2conus path hit beyond ~6 h (deploy_issues #17).

Design: this is exactly the parent
:class:`AWSGRAFRegriddedArchive.open_sample_dataset` body with the
``_open_zarr`` + ``select_time`` pair replaced by a stream+regrid of the
precise frame(s) for the requested step. Everything downstream (destagger,
Raymond filter, diagnostics, level/var subsetting, 5m→15m temporal
aggregation, cloud-cover rescale, valid-time stamping, ``stack_order``) is
the inherited parent logic, run verbatim and in the same order, so the
emitted sample is bit-compatible with the disk-zarr path.

Reuses, rather than re-implements, the streaming + pre-regrid-comp_refl
logic in ``graf2conus`` (``open_operational_init`` with the ``valid_filter``
kwarg) and its ``GRAFRegridder``.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.sources import Source
from ufs2arco.transforms.destagger import destagger
from ufs2arco.transforms.temporal_aggregation import temporal_aggregation

from .graf_utils import compute_reflectivity_3d, apply_raymond_filter_to_dataset
from .local_graf_regridded_operational import LocalGRAFRegriddedOperational

logger = logging.getLogger("ufs2arco")


class GRAFOperationalStreaming(LocalGRAFRegriddedOperational):
    """Per-timestep stream+regrid operational GRAF source.

    Shares all anemoi-side configuration (variables, levels, geographic
    extent, static file, valid-time bookkeeping, Raymond/destagger/temporal
    -aggregation hooks) with :class:`LocalGRAFRegriddedOperational`; only the
    *acquisition* of the per-step regridded field differs.
    """

    # BUCKET is unused (we never read a local zarr) but the parent __init__
    # touches it; a dummy keeps path-builder code from blowing up.
    BUCKET = "/dev/null/_streaming_unused/"

    @property
    def name(self) -> str:
        return "GRAFOperationalStreaming"

    def __init__(
        self,
        file_freqstr: Literal["05m", "15m"],
        init_isos: list[str],
        lead_times: dict,
        graf_s3_base: str,
        weights_dir: str,
        stream_variables: list[str],
        tmp_dir: str = "/var/tmp",
        loader_workers: int = 4,
        target_grid: str = "graf_conus",
        variables: Optional[list] = None,
        static_variables: Optional[list] = None,
        levels: Optional[list | tuple] = None,
        geographic_extent: Optional[dict] = None,
        static_regridded_file_path: Optional[str] = None,
        destagger_kwargs: Optional[dict] = None,
        temporal_aggregation_kwargs: Optional[dict] = None,
        raymond_filter_kwargs: Optional[dict] = None,
        skip_raymond_on_forcing: bool = True,
    ) -> None:
        """Initialize the streaming operational source.

        Parameters
        ----------
        graf_s3_base : str
            Operational bucket root, e.g. ``s3://graf-ai-operational``.
        weights_dir : str
            Directory holding the xESMF weight/grid files + the CONUS
            ``graf_grafconus_geo_mask.npz``.
        stream_variables : list[str]
            Raw GRAF variable names to stream + regrid for this cadence
            (the graf2conus ``variables.nearest`` list for 15m, or
            ``variables.conservative`` for 05m). ``comp_refl`` is computed
            pre-regrid inside graf2conus and may appear here.
        tmp_dir : str
            Scratch dir where graf2conus buffers each streamed NetCDF-5
            (≥ ~12 GB free per concurrent 15m worker).
        loader_workers : int
            Concurrent s5cmd fetch workers inside graf2conus per call. For
            the per-timestep path this is the per-step file count (1 for
            15m, ≤3 for 5m), so keep small.
        Other args: identical to :class:`LocalGRAFRegriddedOperational`.
        """
        # Parent sets up: static file + lat/lon, geo-extent slices, level
        # slices, valid-time bookkeeping (FREQ/forecast_step/n_steps), and the
        # Source grandparent init. It does NOT open any zarr, so the dummy
        # BUCKET is never dereferenced.
        super().__init__(
            file_freqstr=file_freqstr,
            init_isos=init_isos,
            lead_times=lead_times,
            variables=variables,
            static_variables=static_variables,
            levels=levels,
            geographic_extent=geographic_extent,
            static_regridded_file_path=static_regridded_file_path,
            base_dir=self.BUCKET,
            destagger_kwargs=destagger_kwargs,
            temporal_aggregation_kwargs=temporal_aggregation_kwargs,
            raymond_filter_kwargs=raymond_filter_kwargs,
            skip_raymond_on_forcing=skip_raymond_on_forcing,
        )

        self._graf_s3_base = graf_s3_base
        self._weights_dir = weights_dir
        self._tmp_dir = tmp_dir
        self._loader_workers = int(loader_workers)
        self._target_grid = target_grid
        self._stream_variables = list(stream_variables)

        # Lazily constructed per-rank (xESMF weights ~960 MB; build once,
        # reuse across every step this rank handles).
        self._regridder = None
        self._tgt_lats = None
        self._tgt_lons = None
        self._g2c_cfg = self._build_g2c_cfg()

        logger.info(
            "%s: streaming from %s (freq=%s, weights=%s, %d stream vars)",
            self.name, self._graf_s3_base, file_freqstr, weights_dir,
            len(self._stream_variables),
        )

    # ------------------------------------------------------------------ setup
    def _build_g2c_cfg(self):
        """Minimal OmegaConf cfg consumed by graf2conus.open_operational_init."""
        from omegaconf import OmegaConf

        # graf2conus' keep-var union uses both lists; we pass the cadence's
        # variables under the matching key and leave the other empty.
        nearest = self._stream_variables if self.file_freqstr == "15m" else []
        conservative = self._stream_variables if self.file_freqstr != "15m" else []
        return OmegaConf.create(
            {
                "paths": {
                    "graf_s3_base": self._graf_s3_base,
                    "weights_dir": self._weights_dir,
                    "tmp_dir": self._tmp_dir,
                    "loader_workers": self._loader_workers,
                },
                "variables": {"nearest": nearest, "conservative": conservative},
                # No window trim — valid_filter drives file selection.
                "operational": {"lead_time_hours": None, "max_files": None},
            }
        )

    def _ensure_regridder(self):
        if self._regridder is not None:
            return
        from graf2conus.regridder import GRAFRegridder
        from graf2conus.pipeline import _load_target_grid_lats_lons

        self._regridder = GRAFRegridder(self._weights_dir, target=self._target_grid)
        self._tgt_lats, self._tgt_lons = _load_target_grid_lats_lons(
            self._weights_dir, target=self._target_grid
        )

    # ----------------------------------------------------------- acquisition
    def _valid_stamps_for_step(self, init_dt: pd.Timestamp, step: int) -> list[str]:
        """S3 VALID stamps (``YYYYMMDDTHHMMSSZ``) needed for one 15-min step.

        15m: the single frame at ``init + step*15min``.
        05m: the 3 sub-frames ``[3k-2, 3k-1, 3k] * 5min`` that
        :meth:`select_time` / :func:`temporal_aggregation` collapse to one
        15-min bucket. The t=0 window references pre-init frames that do not
        exist on S3; graf2conus' valid_filter tolerates the misses.
        """
        if self.file_freqstr == "15m":
            vts = [init_dt + step * pd.to_timedelta("15min")]
        else:
            five = self.get_5m_steps(step)  # [3k-2, 3k-1, 3k]
            vts = [init_dt + s * pd.to_timedelta("5min") for s in five]
        return [pd.Timestamp(vt).strftime("%Y%m%dT%H%M%SZ") for vt in vts]

    def _stream_and_regrid(self, init_iso: str, valid_stamps: list[str]) -> xr.Dataset:
        """Stream the requested frame(s) and regrid to the (y, x) target grid.

        Returns a dataset shaped exactly like a Time-slice of the old
        ``mpasout_{freq}.zarr`` (dims ``Time``[, ``nVertLevels``]``, y, x`` +
        ``lat``/``lon`` (y,x) coords), ready for ``_prepare_base_xds``.
        """
        from graf2conus.loaders import open_operational_init

        self._ensure_regridder()
        ds = open_operational_init(
            init_iso, self.file_freqstr, self._g2c_cfg,
            valid_filter=set(valid_stamps),
        )
        present = [v for v in self._stream_variables if v in ds.data_vars]
        if not present:
            raise RuntimeError(
                f"{self.name}: none of the stream variables {self._stream_variables} "
                f"present after load (have {list(ds.data_vars)})"
            )
        if self.file_freqstr == "15m":
            reg = self._regridder.regrid_nearest(ds[present])
        else:
            reg = self._regridder.regrid_conservative(ds[present])
        # Mirror AWSGRAFRegriddedArchive._open_zarr: DROP xESMF's lat/lon coords
        # and reset the y/x index coords to bare dims. lat/lon are re-added later
        # as `latitude`/`longitude` DATA VARS by add_static_vars; leaving them as
        # coords here makes ufs2arco's multisource merge (15m + 5m) raise
        # MergeError("unable to determine if {lat, lon} should be coordinates").
        drop = [v for v in ("lat", "lon", "latitude", "longitude") if v in reg.coords]
        if drop:
            reg = reg.drop_vars(drop)
        idx_to_reset = [d for d in ("y", "x") if d in reg.indexes]
        if idx_to_reset:
            reg = reg.reset_index(idx_to_reset, drop=True)
        return reg

    # ----------------------------------------------------------- main entry
    def open_sample_dataset(self, dims, open_static_vars, cache_dir=None):
        """Stream+regrid one step, then run the inherited downstream verbatim."""
        step = int(dims["forecast_step"])
        init_time = dims["init_time"]
        init_dt = self._init_time_dt_map[init_time]

        stamps = self._valid_stamps_for_step(init_dt, step)
        reg = self._stream_and_regrid(init_time, stamps)

        # ---- below mirrors AWSGRAFRegriddedArchive.open_sample_dataset from
        # _prepare_base_xds onward (no _open_zarr / select_time) ----
        xds = self._prepare_base_xds(reg)
        valid_time = self.get_valid_time(xds, init_time=init_time, forecast_step=step)

        # Geographic subset (2D index slices precomputed in __init__).
        xds = xds.isel(y=self._y_slice, x=self._x_slice)

        # Static vars (pre-sliced to geo extent in __init__).
        xds = self.add_static_vars(xds)

        if self.destagger_kwargs is not None:
            xds = destagger(xds, **self.destagger_kwargs)

        # Raymond filter — honor the parent's skip-on-forcing policy: only the
        # IC (step == forecast_offset) is filtered when skip_raymond_on_forcing.
        rk = self.raymond_filter_kwargs
        if self.skip_raymond_on_forcing and step != int(self.forecast_offset):
            rk = None
        if rk is not None:
            xds = apply_raymond_filter_to_dataset(xds, **rk)

        diag_vars = {"comp_refl", "refl_3d", "geopot", "rho", "theta_m"}
        if diag_vars & set(self.variables):
            xds = self.add_diagnostic_variables(xds, self.variables)
            if "refl_3d" in self.variables and "refl_3d" not in xds.data_vars:
                xds = compute_reflectivity_3d(xds)
            xds = xds[self.variables]

        xds = self.apply_slices(xds)
        xds = self.maybe_soil_impute_over_ocean(xds)
        xds = self.maybe_rename_wind_vars(xds)

        if self.temporal_aggregation_kwargs:
            xds = temporal_aggregation(xds, **self.temporal_aggregation_kwargs)

        xds = self.maybe_rescale_cloud_cover(xds)
        xds = self.add_valid_time_to_dataset(xds, valid_time)

        xds.attrs["stack_order"] = ["y", "x"]
        xds.attrs["init_time"] = init_time
        xds.attrs["forecast_step"] = step
        return xds
