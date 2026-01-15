import logging
from typing import Optional
from copy import deepcopy
import re

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from ufs2arco.sources import Source
from ufs2arco.targets import Target

logger = logging.getLogger("ufs2arco")

def _snap_to_valid_time(valid_time, date: pd.Timestamp) -> pd.Timestamp:
    valid_times = pd.to_datetime(valid_time.values)

    if date < valid_times.min() or date > valid_times.max():
        raise ValueError(
            f"Requested date {date} is outside the valid time range: "
            f"{valid_times.min()} to {valid_times.max()}"
        )

    idx = np.argmin(np.abs(valid_times - date))
    return valid_times[idx]

class Anemoi(Target):
    """
    Store dataset ready for anemoi

    Expected output has dimensions
        ``("time", "variable", "ensemble", "cell")``

    Use the rename argument to modify any of these before they get packed in the anemoi dataset.
    This might be useful if you want to train a model with the same variables, but from different datasets,
    so they have different names originally.

    Assumptions:
        * For :class:`EnsembleForecastSource` and :class:`ForecastSource` datasets, t0 gets renamed to time, and fhr is silently dropped
        * :attr:`do_flatten_grid` = ``True``
        * resolution = None, I have no idea where this gets set in anemoi-datasets
        * just setting use_level_index = False for now, but eventually it would be nice to use this flag to switch between how vertical level suffixes are labeled
        * if :attr:`sort_channels_by_level` is ``True``, then we'll make sure that channels go like variable_<level 0> -> variable_<largest level value>
    """

    # these should probably be options
    resolution = None
    use_level_index = False
    allow_nans = True
    data_dtype = np.float32

    # these are basically properties
    always_open_static_vars = True

    @property
    def sample_dims(self):
        if self._has_member:
            return ("time", "ensemble")
        else:
            return ("time",)

    @property
    def expanded_horizontal_dims(self):
        return tuple(self.protected_rename.get(d, d) for d in self.source.horizontal_dims)

    @property
    def horizontal_dims(self):
        if self.do_flatten_grid:
            return ("cell",)
        else:
            return self.expanded_horizontal_dims

    @property 
    def trajectory_ids(self):
        return self.source.trajectory_ids 
        
    @property
    def datetime(self):
        if self._has_fhr:
            return self.source.t0 + pd.Timedelta(hours=self.source.fhr[0])
        else:
            return self.source.valid_times
        
    @property
    def dates(self):
        return self.datetime

    @property
    def time(self):
        # Given the non-unique forecast valid times,
        # best not to rely on the self.datetime property
        # It was causing issues in the datamover.find_my_region func.
        return np.arange(self.source.n_samples)

    @property
    def ensemble(self):
        if self._has_member:
            return self.source.member
        else:
            return [0]

    @property
    def start_date(self):
        return str(self.datetime[0]).replace(" ", "T")

    @property
    def end_date(self):
        return str(self.datetime[-1]).replace(" ", "T")
    
    @property
    def statistics_start_date(self):
        if self.statistics_period.get("start", None) is None:
            return self.start_date
        else:
            date = pd.Timestamp(self.statistics_period.get("start"))
            date = _snap_to_valid_time(self.datetime, date)
            
            assert date in self.datetime, "{self.name}: could not find statistics_start_date within datetime"
            return str(date).replace(" ", "T")

    @property
    def statistics_end_date(self):
        if self.statistics_period.get("end", None) is None:
            return self.end_date
        else:
            date = pd.Timestamp(self.statistics_period.get("end"))
            date = _snap_to_valid_time(self.datetime, date)
            assert date in self.datetime, "{self.name}: could not find statistics_end_date within datetime"
            return str(date).replace(" ", "T")

    @property
    def protected_rename(self) -> dict:
        protected_rename = {
            "latitude": "latitudes",
            "longitude": "longitudes",
        }
        if self._has_fhr:
            protected_rename["valid_time"] = "dates"
        else:
            protected_rename["time"] = "dates"

        if self._has_member:
            protected_rename["member"] = "ensemble"
            
        return protected_rename


    @property
    def dim_order(self):
        return ("time", "variable", "ensemble") + self.horizontal_dims


    def __init__(
        self,
        source: Source,
        chunks: dict,
        store_path: str,
        rename: Optional[dict] = None,
        forcings: Optional[tuple | list] = None,
        statistics_period: Optional[dict] = None,
        compute_temporal_residual_statistics: Optional[bool] = False,
        sort_channels_by_levels: Optional[bool] = False,
        variables_with_nans: Optional[list] = None,
        do_flatten_grid : bool = False
    ) -> None:

        super().__init__(
            source=source,
            chunks=chunks,
            store_path=store_path,
            rename=rename,
            forcings=forcings,
            statistics_period=statistics_period,
            compute_temporal_residual_statistics=compute_temporal_residual_statistics,
        )

        self.sort_channels_by_levels = sort_channels_by_levels
        # additional checks
        if self._has_fhr:
            assert len(self.source.fhr) == 1, \
                f"{self.name}.__init__: Can only use this class with len(fhr)==1, no multiple lead times"

        renamekeys = list(self.rename.keys())
        protected = list(self.protected_rename.keys()) + list(self.protected_rename.values())
        for key in renamekeys:
            if key in protected or self.rename[key] in protected:
                logger.info(f"{self.name}.__init__: can't rename {key} -> {self.rename[key]}, either key or val is in a protected list. I'll drop it and forget about it.")
                self.rename.pop(key)

        self.variables_with_nans = variables_with_nans
        
        self.do_flatten_grid = do_flatten_grid


    def get_expanded_dim_order(self, xds):
        """this is used in :meth:`map_static_to_expanded`"""
        return ("time", "ensemble") + tuple(xds.attrs["stack_order"])


    def apply_transforms_to_sample(
        self,
        xds: xr.Dataset,
    ) -> xr.Dataset:
        
        if self._has_fhr:
            xds["valid_time"] = xds["t0"] + xds["lead_time"].compute()
            xds = xds.squeeze("fhr", drop=True)
            xds = xds.swap_dims({"t0": "valid_time"})
            if "t0" in xds.coords:
                xds = xds.drop_vars("t0")

        if not self._has_member:
            xds = xds.expand_dims({"ensemble": self.ensemble})
            
        # drop ensemble from valid_time
        xds["valid_time"] = xds["valid_time"].squeeze("ensemble")
        
        # In base.py, only 2 transforms are applied
        # (1) renaming , (2) compute forcings 
        xds = super().apply_transforms_to_sample(xds) 
        
        val_time_da = xds["valid_time"]
        lat_da = xds['latitudes']
        lon_da = xds['longitudes']

        xds = self._map_datetime_to_index(xds)
        xds = xds.drop_vars(['latitudes', 'longitudes', 'valid_time'])
        
        xds = self._map_levels_to_suffixes(xds)
        xds = self._map_static_to_expanded(xds)
        dims = ("time", "ensemble") + tuple(xds.attrs.get("stack_order", ()))        
        xds = xds.transpose(*dims)
        xds = self._stackit(xds)
        xds = self._calc_sample_stats(xds)
        if self.do_flatten_grid:
            xds = self._flatten_grid(xds)
        xds = xds.transpose(*self.dim_order)
        xds = xds.reset_coords()
        xds = xds[sorted(xds.data_vars)]
        
        # Add the latitude and longitude back in 
        xds['latitudes'] = lat_da.isel(ensemble=0)
        xds['longitudes'] = lon_da.isel(ensemble=0) 
        
        # Monte: hardcoding the variable dim rechunking.
        xds = xds.chunk({"variable" : -1})

        return xds


    def manage_coords(self, xds: xr.Dataset) -> xr.Dataset:
        attrs = {
            "allow_nans": self.allow_nans,
            "ensemble_dimension": len(self.ensemble),
            "flatten_grid": self.do_flatten_grid,
            "resolution": str(self.resolution),
            "start_date": self.start_date,
            "end_date": self.end_date,
            # Monte: hardcoded!
            "frequency": self.source.STORED_FREQ, 
            "statistics_start_date": self.statistics_start_date,
            "statistics_end_date": self.statistics_end_date,
        }
        xds.attrs.update(attrs)
        return xds


    def rename_dataset(self, xds: xr.Dataset) -> xr.Dataset:
        """
        In addition to any user specified renamings...
        This takes the default source dimensions and renames them to the default anemoi dimensions:

        (t0, member, level, latitude, longitude) -> (dates, ensemble, level, latitudes, longitudes)
        or
        (time, level, latitude, longitude) -> (dates, ensemble, level, latitudes, longitudes)

        Args:
            xds (xr.Dataset): a dataset directly from the source

        Returns:
            xds (xr.Dataset): with renaming as above
        """
        # first, rename the protected list
        # note that we do not update the attribute rename with protected_rename
        # key: values because of the way time is handled ...
        # it is convenient to rename time to dates
        # however... anemoi target has "time" as a logical index in its sample_dims
        # which is completely different...
        # because datamover.create_container relies on Target.renamed_sample_dims attribute,
        # this would rename time to dates, and then create a container with the dates as a dimension...
        # even though it's not
        # then, add on to that the fact that we drop the dates vector and recreate it at the very end,
        # due to the differences in the way xarray and zarr handle datetime objects,
        # we really have to treat these protected quantities differently
        xds = xds.rename(self.protected_rename)
        xds = super().rename_dataset(xds)
        return xds


    def _map_datetime_to_index(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Turn the datetime vector into a logical index and swap the dimensions

        (dates, ensemble, level, latitudes, longitudes) -> (time, ensemble, level, latitudes, longitudes)

        Args:
            xds (xr.Dataset): with time dimension "dates"

        Returns:
            xds (xr.Dataset): with new time dimension "time" ("dates" is still there)
        """
        # Monte: since "time" is a dummy variable for the forecast data, 
        # we want to use "valid_time" here. 
        t = [list(self.datetime).index(date) for date in xds["valid_time"]]
        
        xds["time"] = xr.DataArray(
            t,
            coords=xds["dates"].coords,
            dims=xds["dates"].dims,
            attrs={
                "description": "logical time index",
            },
        )
        xds = xds.swap_dims({"dates": "time"})
        
        # anemoi needs "dates" to be stored as a specific dtype
        # it turns out that this is hard to do consistently with xarray and zarr
        # especially with this "write container" + "fill incrementally" workflow
        # so... let's just store "dates" during aggregate_stats
        if "dates" in xds:
            xds = xds.drop_vars("dates") 
        
        
        return xds

    def _map_levels_to_suffixes(self, xds):
        """
        Take each of the 3D variables, and make them n_levels x 2D variables with the suffix _{level} for each level

        (time, ensemble, level, latitudes, longitudes) -> (time, ensemble, latitudes, longitudes)

        Args:
            xds (xr.Dataset): with all variables, maybe 2D maybe 3D

        Returns:
            nds (xr.Dataset): 3D variables expanded into n_level x 2D variables
        """

        nds = xr.Dataset()
        nds.attrs["variables_metadata"] = dict()

        for name in xds.data_vars:
            meta = {
                "mars": {
                    "date": str(self.datetime[xds.time.values[0]]).replace("-","")[:8],
                    "param": name,
                    "step": 0, # this is the fhr=0 assumption
                    "time": str(self.datetime[xds.time.values[0]]).replace("-","").replace(" ", "").replace(":","")[8:12], # no idea what this should be actually
                    "valid_datetime": str(self.datetime[xds.time.values[0]]).replace(" ", "T"),
                    "variable": name,
                },
            }
            if len(self.ensemble) > 1:
                meta["mars"]["number"] = list(self.ensemble)
            if "level" in xds[name].dims:
                for level in xds[name].level.values:
                    idx = self._get_level_index(xds, level)
                    ilevel = int(level)
                    ilevel = ilevel if ilevel == level else level
                    suffix_name = f"{name}_{ilevel}" if not self.use_level_index else f"{name}_{idx}"
                    nds[suffix_name] = xds[name].sel({"level": level}, drop=True)
                    units = xds["level"].attrs.get('units', '')
                    nds[suffix_name].attrs.update(
                        {
                            "level": ilevel,
                            "level_description": f"{name} at vertical level (index, value) = ({idx}, {ilevel}{units})",
                            "level_index": idx,
                        },
                    )
                    nds.attrs["variables_metadata"][suffix_name] = deepcopy(meta)
                    nds.attrs["variables_metadata"][suffix_name]["mars"]["level"] = ilevel if not self.use_level_index else idx

                if "remapping" not in nds.attrs:
                    nds.attrs["remapping"] = {"param_level": "{param}_{levelist}"}
                else:
                    if "param_level" not in nds.attrs["remapping"].keys():
                        nds.attrs["remapping"]["param_level"] = "{param}_{levelist}"
            else:
                nds[name] = xds[name]

                if "computed_forcing" in xds[name].attrs and "constant_in_time" in xds[name].attrs:
                    nds.attrs["variables_metadata"][name] = {
                        "computed_forcing": xds[name].attrs["computed_forcing"],
                        "constant_in_time": xds[name].attrs["constant_in_time"],
                    }
                else:
                    nds.attrs["variables_metadata"][name] = deepcopy(meta)
                # Is attributes here a hack? Add the "field_shape" here
                # so that it's in the order of the data arrays, not in the dataset order
                # (they could be different)
                if "field_shape" not in nds.attrs and all(d in xds[name].dims for d in self.expanded_horizontal_dims):
                    stack_order = list(d for d in xds[name].dims if d in self.expanded_horizontal_dims)
                    nds.attrs["stack_order"] = stack_order
                    nds.attrs["field_shape"] = list(len(xds[d]) for d in stack_order)

        return nds


    @staticmethod
    def _get_level_index(xds: xr.Dataset, value: int | float) -> int | float:
        return xds["level"].values.tolist().index(value)


    def _sort_channels_by_levels(self, item: tuple) -> tuple:
        """
        If we have a dataset with e.g. geopotential (gh) at 100, 150, 200, ... 1000 hPa
        then the :meth:`_map_levels_to_suffixes` method returns them like this:

            ["gh_100", "gh_1000", "gh_150", ...]

        this method will sort the list of data_vars so they are like this:

            ["gh_100", "gh_150", ... "gh_1000"]

        This is used internally as a key for the sorted function in :meth:`stackit`
        """
        # Match variable names with an underscore followed by a number
        match = re.match(r"(.+)_(\d+)$", item)
        if match:
            var_name, num = match.groups()
            return (var_name, int(num))  # Sort by name, then numeric value

        # Ensure variables like 't2m' are grouped correctly
        return (item, -1)  # Non-numeric suffix variables come before numbered ones


    def _map_static_to_expanded(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Take each variable that does not have any of the (time, ensemble, latitudes, longitudes),
        and expand it so they all look the same

        Args:
            xds (xr.Dataset): with some static variable maybe

        Returns:
            xds (xr.Dataset): all data_vars have the same shape
        """

        for key in xds.data_vars:
            for d in xds.dims:
                if d not in xds[key].dims:
                    xds[key] = xds[key].expand_dims({d: xds[d]})

        return xds


    def _stackit(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Stack the multivariate dataset to a single data array with all variables (and vertical levels) stacked together

        (time, ensemble, latitudes, longitudes) -> (time, ensemble, variable, latitudes, longitudes)

        Args:
            xds (xr.Dataset): with all variables, 3D variables as multiple 2D variables with suffixes

        Returns:
            xds (xr.Dataset): with "data" DataArray, which has all variables/levels stacked together
        """
        varlist = sorted(
            list(xds.data_vars),
            key=self._sort_channels_by_levels if self.sort_channels_by_levels else None,
        )
        channel = [i for i, _ in enumerate(varlist)]
        channel = xr.DataArray(
            channel,
            coords={"variable": channel},
            dims="variable",
        )

        # this might be nice, but it doesn't exist in anemoi
        # and it causes problems with the container / fill workflow
        # it should just be added as a coordinate... but again it's not in anemoi
        #channel_names = xr.DataArray(
        #    varlist,
        #    coords=channel.coords,
        #    dims=channel.dims,
        #)

        data_vars = xr.concat(
            [
                xds[name].expand_dims(
                    {"variable": [this_channel]},
                ).astype(
                    self.data_dtype,
                )
                for this_channel, name in zip(channel, varlist)
            ],
            dim="variable",
            combine_attrs="drop",
        )
        nds = data_vars.to_dataset(name="data")
        nds.attrs = xds.attrs.copy()
        # not making this a data array, even though it might be kinda nice
        nds.attrs["variables"] = varlist
        return nds

    def _flatten_grid(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Flatten (latitudes, longitudes) -> (cell,)

        (time, ensemble, variable, latitudes, longitudes) -> (time, ensemble, variable, cell)

        Args:
            xds (xr.Dataset): with expanded grid

        Returns:
            xds (xr.Dataset): with grid flattened to "cell"
        """
        nds = xds.stack(cell2d=xds.attrs["stack_order"])
        nds["cell"] = xr.DataArray(
            np.arange(len(nds["cell2d"])),
            coords=nds["cell2d"].coords,
            dims=nds["cell2d"].dims,
            attrs={
                "description": f"logical index for 'cell2d', which is a flattened lon x lat array",
            },
        )
        nds = nds.swap_dims({"cell2d": "cell"})

        # For some reason, there's a failure when trying to store this multi-index
        # it's not needed in Anemoi, so no need to keep it anyway.
        nds = nds.drop_vars("cell2d")
        return nds

    def _calc_sample_stats(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Compute statistics for this data sample, which will be aggregated later

        Args:
            xds (xr.Dataset): with just the data and coordinates

        Returns:
            xds (xr.Dataset): with the following statistics, each with an "_array" suffix,
                in order to indicate that the result will still have "time" and "ensemble" dimensions
                that will need to get aggregated
                ["count", "has_nans", "maximum", "minimum", "squares", "sums"]
        """

        dims = list(self.expanded_horizontal_dims)
        xds["count_array"] = (~np.isnan(xds["data"])).sum(dims, skipna=self.allow_nans).astype(np.float64)
        xds["has_nans_array"] = np.isnan(xds["data"]).any(dims)
        xds["maximum_array"] = xds["data"].max(dims, skipna=self.allow_nans).astype(np.float64)
        xds["minimum_array"] = xds["data"].min(dims, skipna=self.allow_nans).astype(np.float64)
        xds["squares_array"] = (xds["data"]**2).sum(dims, skipna=self.allow_nans).astype(np.float64)
        xds["sums_array"] = xds["data"].sum(dims, skipna=self.allow_nans).astype(np.float64)
        return xds


    def finalize(self, topo) -> None:
        """Finalize the dataset with
        * dates
        * stats
        * temporal stats (if specified)
        """

        if topo.is_root:
            self.add_dates()
            self.reconcile_missing_and_nans()
            self.add_trajectory_ids()
            
        topo.barrier()

        logger.info(f"Aggregating statistics")
        self.aggregate_stats(topo)
        logger.info(f"Done aggregating statistics\n")
                
        if self.compute_temporal_residual_statistics:
            logger.info(f"Computing temporal residual statistics")
            self.calc_temporal_residual_stats(topo)
            logger.info(f"Done computing temporal residual statistics\n")
    
    def add_trajectory_ids(self)->None:
        """
        Add the unique trajectory ids for each model init
        """
        xds = xr.open_zarr(self.store_path)
        attrs = xds.attrs.copy()
        
        nds = xr.Dataset()
        nds["trajectory_ids"] = xr.DataArray(
            self.trajectory_ids, 
            coords=xds["time"].coords,
        ) 
        
        nds.attrs = attrs 
        nds.to_zarr(self.store_path, mode="a")
        logger.info(f"{self.name}.add_trajectory_ids: tracjectory ids appended to the dataset\n")
        
        
    def add_dates(self) -> None:
        """Deal with the dates issue

        for some reason, it is a challenge to get the datetime64 dtype to open
        consistently between zarr and xarray, and
        it is much easier to deal with this all at once here
        than in the create_container and incrementally fill workflow.
        """

        xds = xr.open_zarr(self.store_path)
        attrs = xds.attrs.copy()

        nds = xr.Dataset()
        nds["dates"] = xr.DataArray(
            self.datetime.astype("datetime64[s]"),
            coords=xds["time"].coords,
        )
        nds["dates"].encoding = {
            "dtype": "datetime64[s]",
            "units": "seconds since 1970-01-01",
        }

        # store it, first copying the attributes over
        nds.attrs = attrs
        nds.to_zarr(self.store_path, mode="a")
        logger.info(f"{self.name}.add_dates: dates appended to the dataset\n")

    def reconcile_missing_and_nans(self) -> None:
        """This has to happen after :meth:`add_dates` is called.

        Here we do three things:
            1. Make sure missing_dates show up as True in the ``has_nans_array``
               (which propagates to ``has_nans``, since :meth:`aggregate_stats: is called right after this.)
            2. If we have NaNs that we should not have, report it as a missing date
            3. Update missing_indices to match missing_dates for correct forecast data handling
        """

        logger.info(f"{self.name}.reconcile_missing_and_nans: Starting...")

        something_happened = False
        xds = xr.open_zarr(self.store_path)
        xds = xds.swap_dims({"time": "dates"})
        xds["has_nans_array"].load()
        missing_dates = xds.attrs.get("missing_dates", [])
        missing_indices = set(xds.attrs.get("missing_indices", []))
        attrs = xds.attrs.copy()

        # Build a mapping from date to time indices for updating missing_indices
        dates_array = xds["dates"].values
        time_indices = xds["time"].values

        nds = xr.Dataset()
        nds["has_nans_array"] = xds["has_nans_array"]

        # 1. Make sure has_nans_array is True at missing_dates
        logger.info("Checking that has_nans_array = True at each missing_date")
        for mdate in missing_dates:
            try:
                this_one = xds.sel(dates=mdate)
                is_actually_nan = np.isnan(this_one["data"]).any().values
                has_nan = this_one["has_nans_array"].any().values
                if is_actually_nan and not has_nan:
                    something_happened = True
                    logger.info(f" ... setting the date in has_nans_array to True: {mdate}")
                    nds["has_nans_array"].loc[{"dates": mdate}] = True

                # Also ensure index is in missing_indices
                time_idx = int(this_one["time"].values)
                if time_idx not in missing_indices:
                    missing_indices.add(time_idx)
                    something_happened = True
            except KeyError:
                logger.warning(f" ... missing_date {mdate} not found in dates array, skipping")


        # 2. Make sure dates with unexpected NaNs get added to missing_dates
        logger.info("Checking that missing_dates contains all instances of has_nans_array = True (as desired per variable)")
        ignore_idx = []
        if self.variables_with_nans is not None:

            ignoreme = list()
            for ignore_this in self.variables_with_nans:
                if ignore_this in xds.attrs["variables"]:
                    ignoreme.append(ignore_this)
                else:
                    all_instances = [entry for entry in xds.attrs["variables"] if ignore_this in entry]
                    for entry in all_instances:
                        ignoreme.append(entry)

            logger.info(f"Will ignore the following fields if they have NaNs\n{ignoreme}")
            ignore_idx = [xds.attrs["variables"].index(varname) for varname in ignoreme]

        keep_idx = [idx for idx in xds["variable"].values if idx not in ignore_idx]

        nanidx = xds["has_nans_array"].sel(variable=keep_idx).any(["variable", "ensemble"]).values
        nan_time_indices = np.where(nanidx)[0]
        nandates = [str(pd.Timestamp(ndate)) for ndate in xds["dates"][nanidx].values]
        new_missing_dates = list()
        for i, ndate in enumerate(nandates):
            is_missing = ndate in missing_dates
            if not is_missing:
                something_happened = True
                logger.info(f" ... adding date where has_nans_array = True to missing_dates: {ndate}")
                new_missing_dates.append(ndate)

            # Also add to missing_indices
            time_idx = int(nan_time_indices[i])
            if time_idx not in missing_indices:
                missing_indices.add(time_idx)

        if len(new_missing_dates) > 0:
            attrs["missing_dates"] = sorted(missing_dates + new_missing_dates)

        # 3. Update missing_indices in attrs
        attrs["missing_indices"] = sorted(missing_indices)

        if something_happened:
            nds["time"] = xds["time"]
            nds = nds.swap_dims({"dates": "time"}).drop_vars("dates")
            nds.attrs = attrs
            nds.to_zarr(self.store_path, mode="a")
            logger.info(f"{self.name}.reconcile_missing_and_nans: Updated zarr with {len(attrs.get('missing_indices', []))} missing_indices")


    def aggregate_stats(self, topo) -> None:
        """Aggregate statistics over "time" and "ensemble" dimension...
        I'm assuming that this is relatively inexpensive without the spatial dimension

        This will store an array with the statistics

            ["count", "has_nans", "maximum", "mean", "minimum", "squares", "stdev", "sums"]

        and it will get rid of the "_array" versions of the statistics
        """

        xds = xr.open_zarr(self.store_path)
        attrs = xds.attrs.copy()

        # get the start/end times for computing statistics
        # in terms of logical time index values
        start_idx = list(self.datetime).index(pd.Timestamp(self.statistics_start_date))
        end_idx = list(self.datetime).index(pd.Timestamp(self.statistics_end_date))
        xds = xds.sel(time=slice(start_idx, end_idx))

        dims = ["time", "ensemble"]
        time_indices = np.array_split(np.arange(len(xds["time"])), topo.size)
        local_indices = time_indices[topo.rank]

        vidx = xds["variable"].values
        count = np.zeros_like(vidx, dtype=xds["count_array"].dtype)
        has_nans = np.full_like(vidx, fill_value=False, dtype=xds["has_nans_array"].dtype)
        maximum = np.full_like(vidx, fill_value=-np.inf, dtype=xds["maximum_array"].dtype)
        minimum = np.full_like(vidx, fill_value=np.inf, dtype=xds["minimum_array"].dtype)
        squares = np.zeros_like(vidx, dtype=xds["squares_array"].dtype)
        sums = np.zeros_like(vidx, dtype=xds["sums_array"].dtype)

        logger.info(f"{self.name}.aggregate_stats: Performing local computations")
        if local_indices.size > 0:

            lds = xds.isel(time=local_indices)
            local_count = lds["count_array"].sum(dims).compute().values
            local_has_nans = lds["has_nans_array"].any(dims).compute().values
            local_maximum = lds["maximum_array"].max(dims, skipna=True).compute().values
            local_minimum = lds["minimum_array"].min(dims, skipna=True).compute().values
            local_squares = lds["squares_array"].sum(dims, skipna=True).compute().values
            local_sums = lds["sums_array"].sum(dims, skipna=True).compute().values

        else:

            local_count = count.copy()
            local_has_nans = has_nans.copy()
            local_maximum = maximum.copy()
            local_minimum = minimum.copy()
            local_squares = squares.copy()
            local_sums = sums.copy()

        # reduce results
        logger.info(f"{self.name}.aggregate_stats: Communicating results to root")
        topo.sum(local_count, count)
        topo.any(local_has_nans, has_nans)
        topo.max(local_maximum, maximum)
        topo.min(local_minimum, minimum)
        topo.sum(local_squares, squares)
        topo.sum(local_sums, sums)
        logger.info(f"{self.name}.aggregate_stats: ... done communicating")

        # the rest is done on the root rank
        if topo.is_root:
            nds = xr.Dataset()
            kw = {"coords": xds["variable"].coords}
            nds["count"] = xr.DataArray(count, **kw)
            nds["has_nans"] = xr.DataArray(has_nans, **kw)
            nds["maximum"] = xr.DataArray(maximum, **kw)
            nds["minimum"] = xr.DataArray(minimum, **kw)
            nds["squares"] = xr.DataArray(squares, **kw)
            nds["sums"] = xr.DataArray(sums, **kw)

            # now add mean & stdev
            nds["mean"] = nds["sums"] / nds["count"]
            variance = nds["squares"] / nds["count"] - nds["mean"]**2
            nds["stdev"] = xr.where(variance >= 0, np.sqrt(variance), 0.)

            # store it, first copying the attributes over
            nds.attrs = attrs
            nds.to_zarr(self.store_path, mode="a")
            logger.info(f"{self.name}.aggregate_stats: Stored aggregated stats")

        # unclear if this barrier is necessary...
        topo.barrier()


    def calc_temporal_residual_stats(self, topo):

        xds = xr.open_zarr(self.store_path)
        attrs = xds.attrs.copy()
        # Monte: Made this change for my forecast dataset, 
        # which doesn't have a consistent freqstr
        freqstr = xds.attrs['frequency']
        # freqstr = self.dates.freqstr
    
        # get the start/end times for computing statistics
        # in terms of logical time index values
        start_idx = list(self.datetime).index(pd.Timestamp(self.statistics_start_date))
        end_idx = list(self.datetime).index(pd.Timestamp(self.statistics_end_date))
        xds = xds.sel(time=slice(start_idx, end_idx))
        
        # Group the data by the trajectory_ids to avoid cross-forecast leakage. 
        # At the boundaries where the trajectory changes, diffs are masked out. 
        diffs = xds["data"].diff("time")
        same_traj = xds["trajectory_ids"] == xds["trajectory_ids"].shift(time=1)
        diffs = diffs.where(same_traj)
        n_time = len(diffs["time"])

        # Split time indices across ranks
        time_indices = np.array_split(np.arange(n_time), topo.size)
        local_indices = time_indices[topo.rank]

        # Allocate global accumulators
        residual_avg = np.zeros_like(xds["variable"].values, dtype=np.float64)
        residual_var = np.zeros_like(xds["variable"].values, dtype=np.float64)
        residual_max = np.zeros_like(xds["variable"].values, dtype=np.float64)
        residual_min = np.zeros_like(xds["variable"].values, dtype=np.float64)
        count = np.zeros_like(xds["variable"].values, dtype=np.float64)

        logger.info(f"{self.name}.calc_temporal_residual_stats: Performing local computations")
        if local_indices.size > 0:
            dims = [d for d in xds["data"].dims if d != "variable"]
            local_data_diff = diffs.isel(time=local_indices).astype(np.float64)

            local_count = (~np.isnan(local_data_diff)).sum(dims).compute().values.astype(np.float64)
            local_residual_avg = local_data_diff.sum(dims, skipna=True).compute().values
            local_residual_var = (local_data_diff**2).sum(dims, skipna=True).compute().values
            local_residual_max = local_data_diff.max(dims, skipna=True).compute().values
            local_residual_min = local_data_diff.min(dims, skipna=True).compute().values
        else:
            # Return all zeros. 
            local_count = count.copy()
            local_residual_var = residual_var.copy()
            local_residual_avg = residual_avg.copy()
            local_residual_max = residual_max.copy()
            local_residual_min = residual_min.copy()

        logger.info(f"{self.name}.calc_temporal_residual_stats: Communicating results to root")
        topo.sum(local_count, count)
        topo.sum(local_residual_avg, residual_avg)
        topo.sum(local_residual_var, residual_var)
        topo.max(local_residual_max, residual_max)
        topo.min(local_residual_min, residual_min)
        logger.info(f"{self.name}.calc_temporal_residual_stats: ... done communicating")

        if topo.is_root:
            nds = xr.Dataset()
            nds.attrs = attrs

            # Normalize by count
            residual_avg /= count
            residual_var = residual_var / count - residual_avg**2

            ckw = {"coords": xds["variable"].coords}
            nds[f"statistics_tendencies_{freqstr}_mean"] = xr.DataArray(residual_avg, **ckw)
            nds[f"statistics_tendencies_{freqstr}_stdev"] = xr.DataArray(np.sqrt(residual_var), **ckw)
            nds[f"statistics_tendencies_{freqstr}_maximum"] = xr.DataArray(residual_max, **ckw)
            nds[f"statistics_tendencies_{freqstr}_minimum"] = xr.DataArray(residual_min, **ckw)

            nds.to_zarr(self.store_path, mode="a")
            logger.info(f"{self.name}.calc_temporal_residual_stats: Stored temporal residual stats")

        # unclear if this barrier is necessary
        topo.barrier()



    def handle_missing_data(self, missing_data: list[dict]) -> None:
        """Track missing data by storing both missing_dates (for backward compatibility)
        and missing_indices (for correct handling of forecast data with duplicate valid times).

        Note: it is assumed this is only called from the root process

        Args:
            missing_data (list[dict]): list with missing data dicts, containing sample dimensions
                For forecast data: {"init_time": "...", "forecast_step": N}
                For analysis data: {"time": "..."}
        """
        missing_dates = []
        missing_indices = []
        zds = zarr.open(self.store_path, mode="a")

        # Get source properties for computing indices
        has_trajectory_info = (
            hasattr(self.source, 'trajectory_id_dict') and
            hasattr(self.source, 'n_steps') and
            hasattr(self.source, 'forecast_offset')
        )

        for missing_sample in missing_data:
            # Compute valid_time for missing_dates (backward compatibility)
            # Prefer using valid_time if already provided (avoids redundant computation)
            if "valid_time" in missing_sample:
                # Use propagated valid_time from source
                valid_time_str = str(missing_sample["valid_time"])
            elif self._has_fhr:
                valid_time = pd.Timestamp(missing_sample["t0"]) + pd.Timedelta(hours=missing_sample["fhr"])
                valid_time_str = str(valid_time)
            elif "init_time" in missing_sample and "forecast_step" in missing_sample:
                # GRAF-style forecast data - fallback to computation
                init_time = missing_sample["init_time"]
                forecast_step = missing_sample["forecast_step"]
                # Compute valid_time from init_time and forecast_step
                if hasattr(self.source, 'get_valid_time'):
                    valid_time = self.source.get_valid_time(None, init_time, forecast_step)
                    valid_time_str = str(valid_time)
                else:
                    valid_time_str = f"{init_time}_step{forecast_step}"
            else:
                valid_time_str = str(missing_sample.get("time", "unknown"))

            # Add to missing_dates for backward compatibility
            if valid_time_str not in missing_dates:
                missing_dates.append(valid_time_str)

            # Compute exact index for missing_indices
            if "init_time" in missing_sample and "forecast_step" in missing_sample and has_trajectory_info:
                # GRAF-style: compute index from trajectory_id and forecast_step
                init_time = missing_sample["init_time"]
                forecast_step = missing_sample["forecast_step"]
                traj_id = self.source.trajectory_id_dict.get(init_time)
                if traj_id is not None:
                    # Index = trajectory_start + step_within_trajectory
                    idx = traj_id * self.source.n_steps + (forecast_step - self.source.forecast_offset)
                    if idx not in missing_indices:
                        missing_indices.append(idx)
            elif "time" in missing_sample:
                # Analysis data: find index by matching time in dates array
                dates_array = zds["dates"][:]
                target_time = np.datetime64(missing_sample["time"], "s")
                matching = np.where(dates_array == target_time)[0]
                for idx in matching:
                    if int(idx) not in missing_indices:
                        missing_indices.append(int(idx))

        # Store both for compatibility
        zds.attrs["missing_dates"] = missing_dates
        zds.attrs["missing_indices"] = sorted(missing_indices)
        zarr.consolidate_metadata(self.store_path)

        logger.info(f"Stored {len(missing_dates)} missing_dates and {len(missing_indices)} missing_indices")


    def merge_multisource(self, dslist: list[xr.Dataset]) -> xr.Dataset:
        """Take a list of datasets, each from their own source, and merge them"""

        attrs_list = [xds.attrs.copy() for xds in dslist]

        result = xr.concat(dslist, dim="variable", combine_attrs="drop", data_vars="all")

        # these should not have a variable dimension
        for key in ["latitudes", "longitudes"]:
            result[key] = result[key].isel(variable=0, drop=True)

        # create a new variable, to not have [0, 1, 2, 3, 0, 1] or whatever
        result["new_variable"] = xr.DataArray(np.arange(len(result.variable)), coords=result.variable.coords)
        result = result.swap_dims({"variable": "new_variable"}).drop_vars("variable").rename({"new_variable": "variable"})

        result.attrs = _merge_attrs(attrs_list)

        # Rechunk along variable, otherwise this is not worth it!
        result = result.chunk({"variable": self.chunks["variable"]})


        # TODO: resort variable?
        # Or maybe it's more straightforward to leave the order as is, same as concatenating multiple datasets
        return result


def _merge_attrs(list_of_dicts):
    merged_dict = {}
    for d in list_of_dicts:
        for key, value in d.items():
            if key not in ["variables", "variables_metadata", "latest_write_timestamp"]:
                if key in merged_dict and merged_dict[key] != value:
                    raise ValueError(
                        f"Conflict for common key '{key}': "
                        f"Existing value '{merged_dict[key]}' "
                        f"differs from new value '{value}'."
                    )
                merged_dict[key] = value

    # handle these separately
    lists_of_variables = [attrs["variables"] for attrs in list_of_dicts]
    merged_dict["variables"] = [item for sublist in lists_of_variables for item in sublist]

    vmetadata = dict()
    for attrs in list_of_dicts:
        vmetadata.update(attrs["variables_metadata"].copy())

    merged_dict["variables_metadata"] = {key: vmetadata[key] for key in merged_dict["variables"]}
    return merged_dict