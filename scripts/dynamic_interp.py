import xarray as xr
import os
from dask.diagnostics import ProgressBar
from combine_temporal_data import create_zarr_template

workspace = os.path.dirname(os.getcwd())

dynamic_fp = os.path.join(workspace, "data", "combined_output", "dynamic_updated_averaged.zarr")
dynamic = xr.open_zarr(dynamic_fp)

dynamic.rio.set_spatial_dims('lon', 'lat', inplace=True)
dynamic.rio.write_crs("epsg:4326", inplace=True)

interpolated_tiles = []
for t in range(0, dynamic.time.size, 100):
    print(f"Processing time block: {dynamic.time[t].values} to {dynamic.time[t+99].values if t+99 < dynamic.time.size else 'end'}")
    tslice = dynamic.isel(time=slice(t, t+100))
    tslice.rio.set_spatial_dims('lon', 'lat', inplace=True)
    tslice.rio.write_crs("epsg:4326", inplace=True)
    
    # Check for missing values across variables in this block
    need_interp = False
    for var in tslice.data_vars:
        # Compute if any missing values exist in the variable
        if tslice[var].isnull().any().compute():
            need_interp = True
            print(f"  Variable {var} has missing values; interpolating block.")
            break
    if need_interp:
        tslice_interp = tslice.rio.interpolate_na()
    else:
        print("  No missing values in this block; skipping interpolation.")
        tslice_interp = tslice  # use original slice if no NAs
    
    interpolated_tiles.append(tslice_interp)
    
# Concatenate along the time dimension
dynamic_interp = xr.concat(interpolated_tiles, dim="time")
print("Interpolated dynamic data.")

# Now create the Zarr template

lats_10km = dynamic["lat"].values
lons_10km = dynamic["lon"].values
chunk = {"time": 1, "lat": 1, "lon": 1}
compression_level = 5

# Create ag_times from dynamic's time coordinate
ag_times = dynamic["time"].values

output_zarr_dynamic = f"{workspace}/data/combined_output/dynamic_updated_averaged_interp.zarr"

regions_dynamic = create_zarr_template(
    final_path=output_zarr_dynamic,
    var_names=["smaphb_10km"],
    lats=lats_10km,
    lons=lons_10km,
    ag_times=ag_times,
    chunk=chunk,
    compression_level=compression_level,
    additional_attrs={"projection": "EPSG:4326"}
)

print("Dynamic data store template created.")

region_slices = {
    "time": slice(0, len(ag_times)),
    "lat": slice(0, len(lats_10km)),
    "lon": slice(0, len(lons_10km))
}

# Open the template store
ds_zarr = xr.open_zarr(output_zarr_dynamic)

# Remove spatial_ref if present
if "spatial_ref" in ds_zarr:
    ds_zarr = ds_zarr.drop_vars("spatial_ref")
if "spatial_ref" in dynamic_interp:
    dynamic_interp = dynamic_interp.drop_vars("spatial_ref")

dynamic_interp.attrs["projection"] = "EPSG:4326"

with ProgressBar():
    dynamic_interp.to_zarr(
        output_zarr_dynamic,
        region=region_slices,
        mode="a"
    )
