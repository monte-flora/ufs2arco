from .base import Source
from .cloud_zarr import CloudZarrData
from .noaa_grib_forecast import NOAAGribForecastData

# writing something general is actually more work than
# just explicitly writing out the implemented data sources here
_recognized = {
    "aws_gefs_archive": "AWSGEFSArchive",
    "aws_hrrr_archive": "AWSHRRRArchive",
    "gcs_era5_1degree": "GCSERA5OneDegree",
    "gcs_replay_atmosphere": "GCSReplayAtmosphere",
    "gfs_archive": "GFSArchive",
    "aws_aorc": "AWSAORC",
    "aws_graf_archive": "AWSGRAFArchive",
    "aws_graf_regridded_archive": "AWSGRAFRegriddedArchive",
    "aws_graf_regridded_patches": "AWSGRAFRegriddedPatchesArchive",
    "local_graf_regridded_operational": "LocalGRAFRegriddedOperational",
    "graf_operational_streaming": "GRAFOperationalStreaming",
    "wofscast_archive": "WoFSCastArchive",
    "aws_mrms_archive": "AWSMRMSArchive",
    "aws_mrms_patches": "AWSMRMSPatches",
    "aws_hrrr_patches": "AWSHRRRPatches",
    "aws_hrrr_patches_grib2": "AWSHRRRPatchesGrib2",
}

# Lazy imports: source classes are only imported when accessed by name.
# This avoids pulling in heavy/optional dependencies (e.g. grafai -> monte_python)
# for sources that aren't being used.
_lazy_imports = {
    "AWSAORC": ".aws_aorc",
    "AWSGEFSArchive": ".aws_gefs_archive",
    "AWSHRRRArchive": ".aws_hrrr_archive",
    "GCSERA5OneDegree": ".gcs_era5_1degree",
    "GCSReplayAtmosphere": ".gcs_replay_atmosphere",
    "GFSArchive": ".gfs_archive",
    "AWSGRAFArchive": ".aws_graf_reforecast",
    "AWSGRAFRegriddedArchive": ".aws_graf_reforecast_regridded",
    "AWSGRAFRegriddedPatchesArchive": ".aws_graf_regridded_patches",
    "LocalGRAFRegriddedOperational": ".local_graf_regridded_operational",
    "GRAFOperationalStreaming": ".graf_operational_streaming",
    "WoFSCastArchive": ".wofscast",
    "AWSMRMSArchive": ".aws_mrms_archive",
    "AWSMRMSPatches": ".aws_mrms_patches",
    "AWSHRRRPatches": ".aws_hrrr_patches",
    "AWSHRRRPatchesGrib2": ".aws_hrrr_patches_grib2",
}


def __getattr__(name):
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], package=__name__)
        cls = getattr(module, name)
        globals()[name] = cls  # cache for subsequent access
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
