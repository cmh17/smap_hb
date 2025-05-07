import xarray as xr
import os
from dask.diagnostics import ProgressBar
from combine_temporal_data import create_zarr_template

"""
Just adapted from static_interp.py, so need to clean up and probably consolidate
"""
workspace = os.path.dirname(os.getcwd())

# Open target dataset with specified chunks
target = xr.open_zarr(
    f"{workspace}/data/combined_output/target.zarr",
    consolidated=False,
    chunks={'time': 100, 'lat': 360, 'lon': 360}
)

# Set spatial dimensions and CRS
target.rio.set_spatial_dims('lon', 'lat', inplace=True)
target.rio.write_crs("epsg:4326", inplace=True)

# Process interpolation in blocks along time
interpolated_tiles = []
for t in range(0, target.time.size, 100):
    print(f"Processing time block: {target.time[t].values} to {target.time[t+99].values if t+99 < target.time.size else 'end'}")
    tslice = target.isel(time=slice(t, t+100))
    tslice.rio.set_spatial_dims('lon', 'lat', inplace=True)
    tslice.rio.write_crs("epsg:4326", inplace=True)
    
    # Check each data variable in the block for missing values.
    need_interp = False
    for var in tslice.data_vars:
        # Compute if there are any missing values in this variable
        if tslice[var].isnull().any().compute():
            need_interp = True
            print(f"  Variable {var} has missing values; performing interpolation.")
            break
    if need_interp:
        tslice_interp = tslice.rio.interpolate_na()
    else:
        print("  No missing values in this block; skipping interpolation.")
        tslice_interp = tslice  # use original slice
    interpolated_tiles.append(tslice_interp)
    
# Concatenate all the interpolated blocks along the time dimension
target_interp = xr.concat(interpolated_tiles, dim="time")
print("Interpolated target data.")

# Now create the Zarr template
lats_30m = target["lat"].values
lons_30m = target["lon"].values
chunk = {"time": 1, "lat": 360, "lon": 360}
compression_level = 5
ag_times = target["time"].values

output_zarr_target = f"{workspace}/data/combined_output/target_interp_test.zarr"

regions_target = create_zarr_template(
    final_path=output_zarr_target,
    var_names=["smaphb_30m"],
    lats=lats_30m,
    lons=lons_30m,
    ag_times=ag_times,
    chunk=chunk,
    compression_level=compression_level,
    additional_attrs={"projection": "EPSG:4326"}
)

print("Created target data store template.")

region_slices = {
    "time": slice(0, len(ag_times)),
    "lat": slice(0, len(lats_30m)),
    "lon": slice(0, len(lons_30m))
}

# Open the template store and drop spatial_ref if present
ds_zarr = xr.open_zarr(output_zarr_target)
if "spatial_ref" in ds_zarr:
    ds_zarr = ds_zarr.drop_vars("spatial_ref")
if "spatial_ref" in target_interp:
    target_interp = target_interp.drop_vars("spatial_ref")

target_interp.attrs["projection"] = "EPSG:4326"

target_interp = target_interp.chunk({'time': 1})
print("Rechunking complete.")

with ProgressBar():
    target_interp.to_zarr(
        output_zarr_target,
        region=region_slices,
        mode="a"
    )
