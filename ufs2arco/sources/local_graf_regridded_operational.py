# pylint: disable=line-too-long,trailing-whitespace
# Author: monte-flora
# Email: monte.flora@weather.com

"""Operational variant of AWSGRAFRegriddedArchive.

Differences from the training-side parent:
  * Source zarrs are on a local filesystem (NVMe), not S3. ``BUCKET`` is
    repurposed as a local base directory.
  * Init times are provided as an explicit list of ISO strings (e.g.
    ``"2026-05-29T12"``) — there is no GRAF cases CSV at runtime.
  * Per-init zarr paths follow ``{base_dir}/{init_iso}/mpasout_{freq}.zarr``.

Everything else (variable selection, levels, geographic extent, Raymond
filter, destagger, diagnostics, temporal aggregation, time-dim handling)
is inherited unchanged.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from ufs2arco.sources import Source

from .aws_graf_reforecast_regridded import AWSGRAFRegriddedArchive

logger = logging.getLogger("ufs2arco")


class LocalGRAFRegriddedOperational(AWSGRAFRegriddedArchive):
    """Local-NVMe operational source for the per-init regridded GRAF zarrs.

    Consumes the output of ``graf-reforecast-conus-interp``'s operational
    regrid driver (``scripts/run_operational.py``), which produces one
    ``mpasout_{freq}.zarr`` per init time under
    ``{base_dir}/{init_iso}/``.

    Sample dims and downstream processing are identical to the parent:
    DataMover iterates over ``(init_time, forecast_step)``, the same
    diagnostics / Raymond filter / temporal aggregation hooks apply.
    """

    # BUCKET is reused as a base directory (must end with "/"). Parent's
    # _build_path produces "{BUCKET}{case_str}/mpasout_{freq}.zarr", which
    # with case_str=<init_iso> gives the operational layout.
    BUCKET = "/home/mflora/grafai-data/regridded/"

    @property
    def name(self) -> str:
        return "LocalGRAFRegriddedOperational"

    def __init__(
        self,
        file_freqstr: Literal["05m", "15m"],
        init_isos: list[str],
        lead_times: dict[Literal["start", "stop"], str],
        variables: Optional[list] = None,
        static_variables: Optional[list] = None,
        levels: Optional[list | tuple] = None,
        geographic_extent: Optional[dict] = None,
        static_regridded_file_path: Optional[str] = None,
        base_dir: Optional[str] = None,
        destagger_kwargs: Optional[dict] = None,
        temporal_aggregation_kwargs: Optional[dict] = None,
        raymond_filter_kwargs: Optional[dict] = None,
        skip_raymond_on_forcing: bool = True,
    ) -> None:
        """Initialize the operational source.

        Parameters
        ----------
        file_freqstr : "05m" or "15m"
            5-min (accumulated/cumulative vars) or 15-min (prognostic
            vars) source zarr cadence. Match the regridded output.
        init_isos : list[str]
            Explicit init times, ISO format (e.g. ``["2026-05-29T12"]``).
            Each must have a matching directory ``{base_dir}/{iso}/``.
        lead_times : dict
            Same shape as parent: ``{"start": "0h", "stop": "3h"}`` etc.
            Bounds the forecast_step range pulled from each zarr.
        base_dir : str, optional
            Override for ``BUCKET`` — the on-disk root containing per-init
            subdirs. Must end with ``/``. Defaults to the class attribute.
        Other args: as in ``AWSGRAFRegriddedArchive``.
        """
        # Allow per-instance base_dir override; otherwise use the class
        # default. Must end with "/" so parent's f-string path joins work.
        if base_dir is not None:
            if not base_dir.endswith("/"):
                base_dir = base_dir + "/"
            self.BUCKET = base_dir

        self.file_freqstr = file_freqstr
        # Local filesystem — no S3 storage_options needed.
        self._storage_options = {}

        # ---- Build init-time arrays directly from the ISO list ----
        # case_str is just the ISO string itself; parent's path builder
        # treats it as opaque (no CSV lookup needed).
        if not init_isos:
            raise ValueError("init_isos must be a non-empty list of ISO strings")

        self.init_time = list(init_isos)
        # Parse each ISO into a Timestamp. Accepts both "YYYY-MM-DDTHH"
        # and full ISO ("YYYY-MM-DDTHH:MM:SS").
        self._init_time_dt_map = {
            ic: pd.to_datetime(ic) for ic in self.init_time
        }
        self.init_time_dts = pd.to_datetime(
            [self._init_time_dt_map[ic] for ic in self.init_time]
        )
        # init_times_df mimics parent's attr; used by __str__ etc.
        self.init_times_df = pd.DataFrame(
            {"case_str": self.init_time},
            index=self.init_time_dts,
        )
        self.init_times_df.index.name = "init_time"

        self.variables = variables
        self.static_variables = static_variables
        self.levels = levels
        self.lead_times = lead_times
        self.geographic_extent = geographic_extent
        self.static_regridded_file_path = static_regridded_file_path
        self.static_file_path = static_regridded_file_path  # parent __str__ alias
        self.destagger_kwargs = destagger_kwargs
        self.temporal_aggregation_kwargs = temporal_aggregation_kwargs
        self.raymond_filter_kwargs = raymond_filter_kwargs
        self.skip_raymond_on_forcing = bool(skip_raymond_on_forcing)

        # ---- Static file + lat/lon, then optional geo-extent slicing ----
        self._load_static_file_regridded(static_regridded_file_path)
        if self.geographic_extent is not None:
            self._cache_geo_extent_2d()
        else:
            # No subsetting: identity slices over the full (y, x) grid.
            lat = self.lat_lon["latitude"][-1]
            H, W = lat.shape
            self._y_slice = slice(0, H)
            self._x_slice = slice(0, W)

        self._ocean_mask = None

        # ---- Level slices (no cell-based geo slicing on regridded data) ----
        slices: dict = {"isel": {}, "sel": {}}
        if levels:
            slices["isel"]["level"] = levels
            if variables and "smois" in variables:
                slices["isel"]["nSoilLevels"] = 0

        # Initialize via Source (grandparent) — skips parent's CSV loader.
        Source.__init__(
            self, variables, levels=None, use_nearest_levels=False, slices=slices
        )

        # ---- Forecast steps from lead_times bounds ----
        self._compute_valid_times()
        self.forecast_step = (
            np.arange(self.n_steps, dtype=int) + self.forecast_offset
        )

        self._zarr_cache_per_init_time: dict = {}
        self._base_xds_cache_per_init_time: dict = {}

        logger.info(
            "%s: %d init_times × %d steps = %d samples (freq=%s, base_dir=%s, "
            "skip_raymond_on_forcing=%s)",
            self.name, len(self.init_time), self.n_steps,
            len(self.init_time) * self.n_steps, file_freqstr, self.BUCKET,
            self.skip_raymond_on_forcing,
        )

    def open_sample_dataset(self, dims, open_static_vars, cache_dir=None):
        """Skip Raymond on forcing frames (forecast_step > forecast_offset).

        Rationale: anemoi-inference uses only the boundary cells of forcing
        frames (~0.3% of cells via the cutout/boundary mask) to nudge
        the model's rollout; the interior is overwritten by the model's
        own prediction. Smoothing the boundary 5-cell ring with Raymond
        adds zero value to inference quality and costs ~5 CPU-min per
        frame on full-CONUS — a clean ~92% Raymond-cost reduction when
        only the t=0 IC needs training-equivalent filtering.

        Disable with ``skip_raymond_on_forcing=False`` for strict train-
        equivalent A/B runs.
        """
        step = int(dims["forecast_step"])
        is_ic = step == int(self.forecast_offset)
        if self.skip_raymond_on_forcing and not is_ic:
            saved = self.raymond_filter_kwargs
            self.raymond_filter_kwargs = None
            try:
                return super().open_sample_dataset(
                    dims, open_static_vars, cache_dir=cache_dir
                )
            finally:
                self.raymond_filter_kwargs = saved
        return super().open_sample_dataset(
            dims, open_static_vars, cache_dir=cache_dir
        )
