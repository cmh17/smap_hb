import os
import re
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd  # Used to convert date strings to datetime objects

# Set up workspace and paths
workspace = os.getcwd()
daily_path = os.path.join(workspace, "data", "daily")
output_dir = os.path.join(workspace, "data", "imerg", "combined")
Path(output_dir).mkdir(parents=True, exist_ok=True)

# List the day folders
days = os.listdir(daily_path)

# List to store all DataArrays from all files
all_dataarrays = []

# Pattern to extract the date from the filename
pattern = r'cropped_3B-DAY\.MS\.MRG\.(\d{8})-S000000-E235959\.V07B\.nc4'

# Loop through each day folder
for day in days:
    day_folder = os.path.join(daily_path, day)
    if os.path.isdir(day_folder):
        files = os.listdir(day_folder)
        for filename in files:
            match = re.match(pattern, filename)
            if match:
                date_str = match.group(1)  # e.g., "20150401"
                # Convert string to a datetime object
                date = pd.to_datetime(date_str, format='%Y%m%d')
                print(f"Processing file for date {date.date()}")

                # Open the dataset and select the 'precipitation' variable
                ds = xr.open_dataset(os.path.join(day_folder, filename))
                da = ds['precipitation']
                
                # Write the CRS (requires rioxarray)
                da = da.rio.write_crs("EPSG:4326")
                # Replace NaNs with -9999
                da = da.fillna(-9999)
                
                # Ensure there is a time coordinate:
                # If 'time' is not a coordinate, add it.
                if 'time' not in da.coords:
                    da = da.expand_dims("time")
                    da = da.assign_coords(time=date)

                # transpose lat and lon so that lat is first
                da = da.transpose("lat", "lon", "time")

                # Append the DataArray to the list
                all_dataarrays.append(da)

# Concatenate all DataArrays along the 'time' dimension
combined_dataset = xr.concat(all_dataarrays, dim="time")

combined_dataset = combined_dataset.fillna(-9999)
combined_dataset = combined_dataset.rio.write_crs("EPSG:4326")

print(combined_dataset)

# Optionally, save the combined dataset to a NetCDF file
combined_dataset.to_netcdf(os.path.join(output_dir, "combined_imerg.nc"))
