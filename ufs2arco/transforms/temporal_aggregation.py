import xarray as xr 

def temporal_aggregation(
    xds:xr.Dataset,
    reduce_map : dict, 
    resample_kwargs: dict, 
) ->xr.Dataset:
    """
    Using xarray.resample, aggregate data temporally 
    (e.g., reducing 5-min data to a 15-min timestep). 

    Each variable in the dataset is mapped to one or more reduction
    operations (e.g., "sum", "max", "mean") according to the
    `reduce_map`. The function resamples those variables using the
    provided resample frequency and replaces them in the dataset
    with new variables whose names indicate the aggregation applied.

    Parameters
    ----------
    xds : xr.Dataset
        Input dataset containing variables to aggregate.
    reduce_map : dict
        Mapping of reduction name to a list of variable names.
        Example: {"sum": ["precip"], "max": ["temperature"]}.
    resample_kwargs : dict
        Keyword arguments passed to ``xarray.Dataset.resample``,
        typically specifying the resampling dimension and frequency.
        Example: {"time": "15min"}.

    Returns
    -------
    xr.Dataset
        A new dataset with aggregated variables. Original variables
        listed in `reduce_map` are dropped and replaced with new
        variables named ``{var}_{freq}_{stat}``. Attributes
        are preserved, and a new ``long_name`` attribute is assigned
        indicating the aggregation performed.

    Notes
    -----
    - Variables not listed in `reduce_map` remain unchanged.
    - If a variable in `reduce_map` is not present in the dataset,
      it is silently skipped.
    - Works lazily with Dask-backed datasets; computation is deferred
      until explicitly triggered.
    """
    timestr = resample_kwargs["time"]
    
    outs = {}
    for xr_stat, varlist in reduce_map.items():
        varlist = [varlist] if isinstance(varlist, str) else varlist 
        for varname in varlist:
            if varname in xds:
                new_name = f"{varname}_{timestr}_{xr_stat}"
                reduced = getattr(xds[varname].resample(**resample_kwargs), xr_stat)(keep_attrs=True)
                long_name = xds[varname].attrs.get("long_name", varname)
                reduced.attrs["long_name"] = f"{timestr} {xr_stat} of {long_name}"
                outs[new_name] = reduced
                
    return xr.Dataset(outs).transpose("time", ...) 

    
    