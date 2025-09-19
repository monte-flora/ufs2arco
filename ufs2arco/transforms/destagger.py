# Destaggering 
import xarray as xr 
import warnings 

def _drop_attrs(attrs, drop_keys):
    """Helper to drop specified attributes."""
    return {k: v for k, v in attrs.items() if k not in drop_keys}


def _destag_variable(datavar: xr.Variable, stagger_dim=None, unstag_dim_name=None) -> xr.Variable:
    """
    Destaggering function for a single xarray.Variable.
    Based on wrf-python's destagger logic.
    """
    if not isinstance(datavar, xr.Variable):
        raise ValueError(f'Parameter datavar must be xarray.Variable, not {type(datavar)}')

    # Guess the staggered dimension if not provided
    if stagger_dim is not None and stagger_dim not in datavar.dims:
        raise ValueError(f'{stagger_dim} not in {datavar.dims}')
    elif stagger_dim is None:
        stagger_dim = [d for d in datavar.dims if d.endswith('_stag')]
        if len(stagger_dim) > 1:
            raise NotImplementedError(f'Multiple staggered dims found: {stagger_dim}')
        elif not stagger_dim:
            raise ValueError('No staggered dimension found.')
        stagger_dim = stagger_dim[0]

    # Compute mean of adjacent slices
    N = datavar.sizes[stagger_dim]
    left = datavar.isel({stagger_dim: slice(0, N - 1)})
    right = datavar.isel({stagger_dim: slice(1, N)})
    center = 0.5 * (left + right)

    # Rename the dimension
    if unstag_dim_name is None:
        unstag_dim_name = stagger_dim.replace('_stag', '')

    new_dims = tuple(unstag_dim_name if d == stagger_dim else d for d in center.dims)

    return xr.Variable(
        dims=new_dims,
        data=center.data,
        attrs=_drop_attrs(datavar.attrs, ('stagger', 'c_grid_axis_shift')),
        encoding=datavar.encoding,
        fastpath=True,
    )

def destagger(xds: xr.Dataset, vars_to_destag : dict, unstag_dim_names : dict = None ) -> xr.Dataset:
    """
    Efficiently destagger selected variables in the dataset using the stagger map.
    This version uses modular destaggering logic per-variable.
    
    Inspired by https://github.com/xarray-contrib/xwrf/blob/main/xwrf/destagger.py
    
    Args:
        xds : xr.Dataset
        vars_to_destag : dict 
            Pairs of variables to destagger and the staggered dimension 
            e.g., {"w" : "bottom_top_staggered"} 
        unstag_dim_names : dict
            Pairs of staggered dimension names and the new unstaggered names. 
            e.g., {"bottom_top_staggered" : "level"}
            Defaults to None, which will not change the dim name. 
    """
    if unstag_dim_names is None:
        unstag_dim_names = {}
    

    destagged_vars = {}
    for var, stagger_dim in vars_to_destag.items():

        variable = xds[var].variable  # get the xarray.Variable from DataArray
        if stagger_dim not in unstag_dim_names:
            warnings.warn(
                f"No unstaggered name provided for dim '{stagger_dim}'. "
                f"Using original name.", 
                UserWarning
            )
        unstag_dim_name = unstag_dim_names.get(stagger_dim, stagger_dim)
    
        destagged_var = _destag_variable(variable, 
                                             stagger_dim=stagger_dim, 
                                             unstag_dim_name=unstag_dim_name)
        destagged_vars[var] = xr.DataArray(destagged_var, coords=xds[var].coords)

    return xds.drop_vars(destagged_vars.keys()).assign(destagged_vars)
