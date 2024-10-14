import os
import re
from pathlib import Path
import xarray as xr
import numpy as np

workspace = os.getcwd()
print(workspace)

base_dir = "%s/data/smaphb/SMAPHB_sample/" % workspace
output_dir = "%s/data/daily/" % workspace

Path(output_dir).mkdir(parents=True, exist_ok=True)

# Pattern to match filenames in the format 'SMAP-HB_surface-soil-moisture_30m_daily_YYYY-MM.nc'
pattern = r'SMAP-HB_surface-soil-moisture_30m_daily_(\d{4})-(\d{2})\.nc'

# Threshold for minimum valid data points (80% valid data)
valid_threshold = 0.8

for filename in os.listdir(base_dir):
    if filename.endswith(".nc"):
        match = re.match(pattern, filename)
        if match:
            year = match.group(1)
            month = match.group(2)
            
            ds = xr.open_dataset(os.path.join(base_dir, filename))

            for i in range(len(ds["time"])):
                day_str = f"{ds['time'][i].dt.day:02d}"  # Format the day with leading zeros (e.g., '01', '02')
                
                # Select data for this day
                this_day = ds.isel(time=i)
                
                # Calculate the percentage of valid data points (non-NaN values)
                total_points = np.prod(this_day['SMAPHB_SM'].shape)  # Total number of points in the raster
                valid_points = np.count_nonzero(~np.isnan(this_day['SMAPHB_SM']))  # Non-NaN points
                
                valid_percentage = valid_points / total_points
                
                # Only create the folder and save the file if the valid percentage exceeds the threshold
                if valid_percentage >= valid_threshold:
                    # Construct the daily folder path
                    day_folder = os.path.join(output_dir, f"{year}-{month}-{day_str}")
                    Path(day_folder).mkdir(parents=True, exist_ok=True)
                    
                    # Save the NetCDF for this day, overwriting if it already exists
                    this_day.to_netcdf(os.path.join(day_folder, f"SMAPHB_SM_{year}-{month}-{day_str}.nc"), mode="w")
                    print(f"Saved {day_folder} with {valid_percentage:.2%} valid data")
                else:
                    print(f"Skipping {year}-{month}-{day_str}: Only {valid_percentage:.2%} valid data")

        else:
            print(f"Filename does not match expected format: {filename}")
