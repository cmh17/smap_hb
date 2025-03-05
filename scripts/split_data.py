import xarray as xr
import os

workspace = os.getcwd()

# Open datasets
static = xr.open_dataset(f"{workspace}/data/combined_output/combined_static_data.nc")  # includes DEM, Polaris, etc.
dynamic = xr.open_zarr(f"{workspace}/data/combined_output/dynamic.zarr", consolidated=False)  # includes SMAP 10km, IMERG 10km
target = xr.open_zarr(f"{workspace}/data/combined_output/target.zarr", consolidated=False)  # includes SMAP 30 m

# Create output folder for tiles if it doesn't exist
tiles_dir = os.path.join(workspace, "data/combined_output/tiles")
os.makedirs(tiles_dir, exist_ok=True)

# Split the data into 10x10 tiles (each tile covering a specific range)
for i in range(10):
    for j in range(10):
        print(f"Processing tile {i}, {j}")
        tile_dir = os.path.join(tiles_dir, f"tile_{i}_{j}")
        os.makedirs(tile_dir, exist_ok=True)
        
        # Define file paths for outputs
        static_file = os.path.join(tile_dir, "static_predictors.nc")
        dynamic_file = os.path.join(tile_dir, "dynamic_predictors.nc")
        target_file = os.path.join(tile_dir, "target.nc")
        
        # Process static tile if file doesn't exist
        if not os.path.exists(static_file):
            static_tile = static.isel(
                lat=slice(i*360, (i+1)*360),
                lon=slice(j*360, (j+1)*360)
            )
            static_tile.to_netcdf(static_file)
        else:
            print(f"  Skipping static predictors for tile {i}, {j} (file exists)")
        
        # Process dynamic tile if file doesn't exist
        if not os.path.exists(dynamic_file):
            dynamic_tile = dynamic.isel(
                lat_10km=slice(i, i+1),
                lon_10km=slice(j, j+1)
            )
            dynamic_tile.to_netcdf(dynamic_file)
        else:
            print(f"  Skipping dynamic predictors for tile {i}, {j} (file exists)")
        
        # Process target tile if file doesn't exist
        if not os.path.exists(target_file):
            target_tile = target.isel(
                lat_30m=slice(i*360, (i+1)*360),
                lon_30m=slice(j*360, (j+1)*360)
            )
            target_tile.to_netcdf(target_file)
        else:
            print(f"  Skipping target for tile {i}, {j} (file exists)")

print("Data split into 100 tiles.")
