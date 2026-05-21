import gzip
import logging
import os
import shutil
import tempfile
from typing import Optional

import pandas as pd
import s3fs
import xarray as xr
import yaml

from ufs2arco.sources import Source

logger = logging.getLogger("ufs2arco")

_BUCKET = "noaa-mrms-pds"
_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"


class AWSMRMSArchive(Source):
    """
    Access MRMS MultiSensor_QPE_01H_Pass2_00.00 from the NOAA AWS open-data bucket.

    S3 path pattern:
        s3://noaa-mrms-pds/CONUS/MultiSensor_QPE_01H_Pass2_00.00/YYYYMMDD/
            MRMS_MultiSensor_QPE_01H_Pass2_00.00_YYYYMMDD-HHMMSS.grib2.gz

    Files are GRIB2 gzipped; cfgrib decodes the QPE field as 'unknown' because MRMS
    uses a custom NOAA local parameter table (paramId=0). We rename it to 'qpe_01h'.

    Units: mm (1-hour QPE accumulation ending at valid_time).

    Note: no explicit regridding is performed. The native MRMS lat/lon grid
    (3500 × 7000, 0.01°) is returned as-is.
    """

    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")
    static_vars = ()
    available_levels = ()

    @property
    def available_variables(self) -> tuple:
        return tuple(self._varmeta.keys())

    @property
    def rename(self) -> dict:
        return {}

    def __init__(
        self,
        time: dict,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
    ) -> None:
        """
        Args:
            time (dict): Passed to ``pandas.date_range`` — keys: start, end, freq.
                         Example: {"start": "2021-01-01", "end": "2021-01-31", "freq": "1h"}
            variables (list, tuple, optional): variables to grab (default: all)
            levels (list, tuple, optional): unused for MRMS (surface-only product)
            use_nearest_levels (bool, optional): unused
            slices (dict, optional): xarray sel/isel slices to apply
        """
        ref_path = os.path.join(os.path.dirname(__file__), "reference.mrms.yaml")
        with open(ref_path) as f:
            self._varmeta = yaml.safe_load(f)

        self.time = pd.date_range(**time)

        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
        )

    def _build_path(self, time: pd.Timestamp) -> str:
        date = f"{time.year:04d}{time.month:02d}{time.day:02d}"
        hms = f"{time.hour:02d}0000"
        fname = f"MRMS_{_PRODUCT}_{date}-{hms}.grib2.gz"
        return f"s3://{_BUCKET}/CONUS/{_PRODUCT}/{date}/{fname}"

    def open_sample_dataset(
        self,
        dims: dict,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:
        """
        Open one MRMS QPE file for the given valid time.

        Args:
            dims (dict): Must contain key 'time' as a pd.Timestamp.
            open_static_vars (bool): ignored (MRMS has no static vars)
            cache_dir (str, optional): local directory to cache downloaded .gz files

        Returns:
            xr.Dataset with variable 'qpe_01h' and dims (time, latitude, longitude).
            Returns empty xr.Dataset on failure.
        """
        t = dims["time"]
        s3_path = self._build_path(t)
        logger.debug(f"{self.name}.open_sample_dataset: reading {s3_path}")

        tmp_grib = None
        try:
            tmp_grib = self._fetch_and_decompress(s3_path, cache_dir)
            if tmp_grib is None:
                return xr.Dataset()

            xds = self._open_grib(tmp_grib)
            if xds is None:
                return xr.Dataset()

            xds = self._standardize(xds, t)
            if self.slices:
                xds = self.apply_slices(xds)
            return xds[self.variables]

        except Exception as e:
            logger.warning(f"{self.name}: failed to open {s3_path}: {e}")
            return xr.Dataset()

        finally:
            if tmp_grib and os.path.exists(tmp_grib):
                os.unlink(tmp_grib)

    def _fetch_and_decompress(self, s3_path: str, cache_dir: Optional[str]) -> Optional[str]:
        """Download .grib2.gz from S3 (optionally caching the compressed file),
        decompress to a temp .grib2, and return the temp file path."""
        fs = s3fs.S3FileSystem(anon=True)

        # path without s3:// prefix for s3fs
        s3_key = s3_path.removeprefix("s3://")

        # optionally cache the .gz file
        gz_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            gz_path = os.path.join(cache_dir, os.path.basename(s3_key))

        if gz_path and os.path.exists(gz_path):
            logger.debug(f"{self.name}: using cached {gz_path}")
        else:
            try:
                if gz_path:
                    fs.get(s3_key, gz_path)
                    src = gz_path
                else:
                    # stream directly without caching
                    with tempfile.NamedTemporaryFile(suffix=".grib2.gz", delete=False) as tmp:
                        gz_path_tmp = tmp.name
                    fs.get(s3_key, gz_path_tmp)
                    gz_path = gz_path_tmp
            except Exception as e:
                logger.warning(f"{self.name}: S3 download failed for {s3_path}: {e}")
                return None

        # decompress to temp .grib2
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp_grib = tmp.name

        try:
            with gzip.open(gz_path, "rb") as f_in, open(tmp_grib, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            logger.warning(f"{self.name}: decompression failed for {gz_path}: {e}")
            os.unlink(tmp_grib)
            return None
        finally:
            # remove temp gz if we didn't cache it explicitly
            if cache_dir is None and gz_path:
                try:
                    os.unlink(gz_path)
                except OSError:
                    pass

        return tmp_grib

    def _open_grib(self, grib_path: str) -> Optional[xr.Dataset]:
        """Open the decompressed GRIB2 file with cfgrib."""
        fbk = self._varmeta["qpe_01h"]["filter_by_keys"].copy()
        try:
            xds = xr.open_dataset(
                grib_path,
                engine="cfgrib",
                filter_by_keys=fbk,
                backend_kwargs={"indexpath": ""},
                decode_timedelta=True,
            )
        except Exception as e:
            logger.warning(f"{self.name}: cfgrib failed on {grib_path}: {e}")
            return None

        # rename 'unknown' → 'qpe_01h'
        og = self._varmeta["qpe_01h"]["original_name"]
        if og in xds.data_vars:
            xds = xds.rename({og: "qpe_01h"})
        xds["qpe_01h"].attrs["long_name"] = self._varmeta["qpe_01h"]["long_name"]
        xds["qpe_01h"].attrs["units"] = self._varmeta["qpe_01h"]["units"]

        return xds

    def _standardize(self, xds: xr.Dataset, t: pd.Timestamp) -> xr.Dataset:
        """Standardize coordinates: add time dim, drop scalar coords."""
        # MRMS valid_time == time; use the requested timestamp as the canonical time
        for coord in ("step", "heightAboveSea", "time", "valid_time"):
            if coord in xds.coords and coord not in xds.dims:
                xds = xds.drop_vars(coord)

        # add explicit time dimension
        xds = xds.expand_dims(dim={"time": [t]})
        xds["time"].attrs = {
            "long_name": "valid time (UTC)",
            "description": "End time of 1-hr QPE accumulation window",
        }
        return xds
