#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import xarray as xr
import numpy as np
import rioxarray
from rasterio.enums import Resampling

"""
Resample the upscaled 50 km-resolution SMAP data so that they stay low-res but their arrays have the same dimensions as everything else (3600 x 3600).
"""

# Set up the base directory where the 'daily' folder is located
base_dir = os.path.join(os.getcwd(), 'data', 'daily')

# Loop through each date folder in the 'daily' directory
for date_folder in os.listdir(base_dir):
    # Define the path to the current date folder
    date_folder_path = os.path.join(base_dir, date_folder)
    
    # Check if it's indeed a directory (to skip files, if any)
    if os.path.isdir(date_folder_path):
        print(f"Processing folder: {date_folder_path}")

        if os.path.exists(output_file):
            print(f"Skipping {date_folder_path}, resampled output already exists.")
            continue  # Skip this directory and move to the next one
        
        # Loop through files in the date folder and process only SMAP files
        for file_name in os.listdir(date_folder_path):
            if file_name.startswith('SMAP') and file_name.endswith('_50km.nc'):  # Assuming SMAP files follow this naming pattern
                # Define the full path to the input SMAP file
                input_file = os.path.join(date_folder_path, file_name)
                
                # Define the output file name (resampled data)
                output_file = os.path.join(date_folder_path, f"SMAP_{date_folder}_resampled.nc")
                
                # Load the SMAP dataset
                ds_smap = xr.open_dataset(input_file)

                # Drop non-spatial variables like 'time_bnds' to avoid errors (if applicable)
                ds_smap_clean = ds_smap.drop_vars('time_bnds', errors='ignore')

                # Assign CRS (coordinate reference system)
                ds_smap_clean = ds_smap_clean.rio.write_crs("EPSG:4326", inplace=True)

                # Rename lon/lat to x/y for rioxarray to recognize them as spatial dimensions
                ds_smap_clean = ds_smap_clean.rename({'lon': 'x', 'lat': 'y'})

                # Transpose the data to the correct order ('time', 'y', 'x')
                ds_smap_clean = ds_smap_clean.transpose('time', 'y', 'x')

                # Set spatial dimensions explicitly for rioxarray
                ds_smap_clean = ds_smap_clean.rio.set_spatial_dims(x_dim='x', y_dim='y')

                # Get the shape and resolution from IMERG resampled files (assuming same target resolution)
                target_resolution = 30 / (111320 * np.cos(np.deg2rad(30)))  # Match IMERG target resolution
                
                # Match the shape of the SMAP data to IMERG (assuming similar grid size as IMERG)
                ds_smap_resampled = ds_smap_clean.rio.reproject(
                    ds_smap_clean.rio.crs, 
                    shape=(3600, 3600),  # Same shape as IMERG resampled (adjust as needed)
                    resampling=Resampling.nearest  # Adjust the resampling method if necessary (nearest, bilinear, etc.)
                )

                # Save the resampled dataset
                ds_smap_resampled.to_netcdf(output_file)
                print(f"Resampled SMAP data saved to: {output_file}")
