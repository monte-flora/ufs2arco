import xarray as xr 

def temporal_aggregation(
    xds: xr.Dataset,
    reduce_map: dict,
    time_coord: xr.DataArray | None = None,
) -> xr.Dataset:
    """
    Perform temporal aggregation over the time dimension,
    preserving latitude/longitude and keeping a length-1 time dimension.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset with a time dimension (e.g., 5-min slices)
    reduce_map : dict
        {"sum": ["apcp"], "max": ["reflectivity"], ...}
    time_coord : xr.DataArray, optional
        Time coordinate to use for the aggregated output.
        If None, uses the last time in xds.
    """

    if "time" not in xds.dims:
        raise ValueError("temporal_aggregation requires a 'time' dimension")

    # Determine output time coordinate
    if time_coord is None:
        time_coord = xds["time"].isel(time=-1)

    out_vars = {}

    # ------------------------------------------------------------------
    # Aggregate requested variables
    # ------------------------------------------------------------------
    for stat, varlist in reduce_map.items():
        varlist = [varlist] if isinstance(varlist, str) else varlist

        for varname in varlist:
            if varname not in xds:
                continue

            da = xds[varname]

            if stat == "sum":
                reduced = da.sum(dim="time", keep_attrs=True)
            elif stat == "mean":
                reduced = da.mean(dim="time", keep_attrs=True)
            elif stat == "max":
                reduced = da.max(dim="time", keep_attrs=True)
            elif stat == "min":
                reduced = da.min(dim="time", keep_attrs=True)
            else:
                reduced = da.reduce(getattr(np, stat), dim="time", keep_attrs=True)

            new_name = f"{varname}_15min_{stat}"
            reduced = reduced.rename(new_name)

            long_name = da.attrs.get("long_name", varname)
            reduced.attrs["long_name"] = f"15-min {stat} of {long_name}"

            #  restore time dimension (length 1)
            reduced = reduced.expand_dims(time=[time_coord.values])

            out_vars[new_name] = reduced

    # ------------------------------------------------------------------
    # Pass through non-time-dependent variables (lat, lon, masks, etc.)
    # ------------------------------------------------------------------
    passthrough = {
        v: xds[v]
        for v in xds.data_vars
        if "time" not in xds[v].dims
    }

    # ------------------------------------------------------------------
    # Build output dataset
    # ------------------------------------------------------------------
    out = xr.Dataset(
        data_vars={**out_vars, **passthrough},
        coords={
            "time": ("time", [time_coord.values]),
            **{k: v for k, v in xds.coords.items() if k != "time"},
        },
        attrs=xds.attrs,
    )

    return out



