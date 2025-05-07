import xarray as xr
import os
from dask.diagnostics import ProgressBar

from static_zarr import create_static_zarr_template

# Copied this out of a notebook, so need to clean up and check
workspace = os.path.dirname(os.getcwd())

static_fp = os.path.join(workspace, "data", "combined_output", "static.zarr")
static = xr.open_zarr(static_fp, consolidated=False, chunks={'lat': 360, 'lon': 360})


# Set spatial dims and coords
static.rio.set_spatial_dims('lon', 'lat', inplace=True)
static.rio.write_crs("epsg:4326", inplace=True)

for var in static.data_vars:
    print(f"Processing {var}")
    da = static[var]
    da = da.rio.set_spatial_dims('lon', 'lat', inplace=False)
    da = da.rio.write_crs("epsg:4326", inplace=False)
    
    # Check for any missing values; compute the result since it's lazy
    has_nas = da.isnull().any().compute()
    
    if has_nas:
        print(f"  Missing values found in {var}; interpolating...")
        da = da.rio.interpolate_na()  # Use the default method (nearest)
    else:
        print(f"  No missing values for {var}; skipping interpolation.")
        
    static[var] = da

print("Interpolated static data")

# final lat/lon arrays
lats = static["lat"].values
lons = static["lon"].values

# Include all vars for now
var_names = list(static.data_vars)

chunk = {"lat": 360, "lon": 360}
compression_level = 5
output_zarr_dir = f"{workspace}/data/combined_output/static_interp.zarr"

create_static_zarr_template(
    final_path=output_zarr_dir,
    var_names=var_names,
    lats=lats,
    lons=lons,
    chunk=chunk,
    compression_level=compression_level,
    additional_attrs={"projection": "EPSG:4326"}
)

# Write the final data into the template using region slicing
region_slices = {
    "lat": slice(0, len(lats)),
    "lon": slice(0, len(lons))
}

# open the template store
ds_zarr = xr.open_zarr(output_zarr_dir)

# remove spatial_ref from the template
if "spatial_ref" in ds_zarr:
    ds_zarr = ds_zarr.drop_vars("spatial_ref")

static = static.drop_vars("spatial_ref")

static.attrs["projection"] = "EPSG:4326"

with ProgressBar():
    static.to_zarr(
        output_zarr_dir,
        region=region_slices,
        mode="a"
    )
