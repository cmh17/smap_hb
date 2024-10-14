import os
import xarray as xr

# Set up working directories
workspace = os.getcwd()
input_dir = f"{workspace}/data/daily"

# Loop through each NetCDF file in the input directory (and subdirectories)
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".nc") and filename.startswith("SMAPHB"):
            input_path = os.path.join(root, filename)
            
            output_filename = filename.replace(".nc", "_50km.nc")
            output_path = os.path.join(root, output_filename)

            ds = xr.open_dataset(input_path)

            factor = 1667  # Number of 30m pixels per 50km side
            ds_downscaled = ds.coarsen(lon=factor, lat=factor, boundary='pad').mean()

            ds_downscaled.to_netcdf(output_path)

            print(f"Downscaled file saved: {output_path}")
