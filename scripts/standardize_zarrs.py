import os
import xarray as xr
from static_zarr import create_static_zarr_template # Should probably rename this script to be more descriptive
from combine_temporal_data import create_zarr_template
from dask.diagnostics import ProgressBar

workspace = os.path.dirname(os.getcwd())

# target = xr.open_zarr(f"{workspace}/data/combined_output/target_interp.zarr")
static = xr.open_zarr(f"{workspace}/data/combined_output/static_interp.zarr")
dynamic = xr.open_zarr(f"{workspace}/data/combined_output/dynamic_updated_averaged_interp.zarr")

# print(target["smaphb_30m"].chunks)
# print(static.commercial.chunks)
# print(dynamic.smaphb_10km.chunks)
# print(dynamic.imerg_10km.chunks)

# # Do it for the target data
# for var in target.data_vars: # It's just one var
#     print(f"Processing {var}")
#     da = target[var]
#     da_std = (da - da.mean())/da.std()
        
#     target[var] = da_std
    
# Do it for the static data, all vars
for var in static.data_vars:
    print(f"Processing {var}")
    da = static[var]
    da_std = (da - da.mean())/da.std()
        
    static[var] = da_std
    
# Do it for the dynamic data, both vars
for var in dynamic.data_vars:
    print(f"Processing {var}")
    da = dynamic[var]
    da_std = (da - da.mean())/da.std()
        
    dynamic[var] = da_std
    
    
# Save
# First need to create zarr templates for target, static, and dynamic

# lats_30m = target["lat"].values
# lons_30m = target["lon"].values
# chunk = {"time": 1, "lat": 360, "lon": 360}
# compression_level = 5
# ag_times = target["time"].values

# output_zarr_target = f"{workspace}/data/combined_output/target_interp_std.zarr"

# regions_target = create_zarr_template(
#     final_path=output_zarr_target,
#     var_names=target.data_vars,
#     lats=lats_30m,
#     lons=lons_30m,
#     ag_times=ag_times,
#     chunk=chunk,
#     compression_level=compression_level,
#     additional_attrs={"projection": "EPSG:4326"}
# )

# print("Created target data store template.")

# region_slices = {
#     "time": slice(0, len(ag_times)),
#     "lat": slice(0, len(lats_30m)),
#     "lon": slice(0, len(lons_30m))
# }

# # Open the template store and drop spatial_ref if present
# ds_zarr = xr.open_zarr(output_zarr_target)
# if "spatial_ref" in ds_zarr:
#     ds_zarr = ds_zarr.drop_vars("spatial_ref")
# if "spatial_ref" in target:
#     target = target.drop_vars("spatial_ref")

# # Save projection as an attribute instead
# target.attrs["projection"] = "EPSG:4326"

# target = target.chunk({'time': 1})
# print("Rechunking complete.")

# with ProgressBar():
#     target.to_zarr(
#         output_zarr_target,
#         region=region_slices,
#         mode="a"
#     )

# print("Standardized target saved.")

# Now do the same for static data
# final lat/lon arrays
lats = static["lat"].values
lons = static["lon"].values

# Include all vars for now
var_names = list(static.data_vars)

chunk = {"lat": 360, "lon": 360}
compression_level = 5
output_zarr_dir = f"{workspace}/data/combined_output/static_interp_std.zarr"

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

if "spatial_ref" in static:
    static = static.drop_vars("spatial_ref")

static.attrs["projection"] = "EPSG:4326"

with ProgressBar():
    static.to_zarr(
        output_zarr_dir,
        region=region_slices,
        mode="a"
    )
    
print("Saved standardized static data.")

# Dynamic
lats_10km = dynamic["lat"].values
lons_10km = dynamic["lon"].values
chunk = {"time": 1, "lat": 1, "lon": 1}
compression_level = 5

# Create ag_times from dynamic's time coordinate
ag_times = dynamic["time"].values

output_zarr_dynamic = f"{workspace}/data/combined_output/dynamic_updated_averaged_interp_std.zarr" # That's a long name...

regions_dynamic = create_zarr_template(
    final_path=output_zarr_dynamic,
    var_names=list(dynamic.data_vars),
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
if "spatial_ref" in dynamic:
    dynamic = dynamic.drop_vars("spatial_ref")

dynamic.attrs["projection"] = "EPSG:4326"

with ProgressBar():
    dynamic.to_zarr(
        output_zarr_dynamic,
        region=region_slices,
        mode="a"
    )