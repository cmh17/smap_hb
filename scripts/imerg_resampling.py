#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script performs resampling of IMERG data and compares original vs resampled data.
Author: Carrie Hashimoto
Date: 2024-10-13
"""

import os
import xarray as xr
import numpy as np
import rioxarray
from rasterio.enums import Resampling

base_dir = os.path.join(os.getcwd(), 'data', 'daily')

for date_folder in os.listdir(base_dir):

    date_folder_path = os.path.join(base_dir, date_folder)
    
    if os.path.isdir(date_folder_path):
        print(f"Processing folder: {date_folder_path}")
        
        output_file = os.path.join(date_folder_path, f"IMERG_{date_folder}_resampled_30m.nc")
        
        if os.path.exists(output_file):
            print(f"Skipping {date_folder_path}, resampled output already exists.")
            continue 
        
        for file_name in os.listdir(date_folder_path):
            if file_name.startswith('cropped_3B-DAY.MS.MRG') and file_name.endswith('.nc4'):

                input_file = os.path.join(date_folder_path, file_name)
                
                ds_imerg = xr.open_dataset(input_file)

                ds_imerg_clean = ds_imerg.drop_vars('time_bnds', errors='ignore')

                ds_imerg_clean = ds_imerg_clean.rio.write_crs("EPSG:4326", inplace=True)

                ds_imerg_clean = ds_imerg_clean.rename({'lon': 'x', 'lat': 'y'})

                ds_imerg_clean = ds_imerg_clean.transpose('time', 'y', 'x')

                ds_imerg_clean = ds_imerg_clean.rio.set_spatial_dims(x_dim='x', y_dim='y')

                target_resolution = 30 / (111320 * np.cos(np.deg2rad(30)))

                ds_imerg_resampled = ds_imerg_clean.rio.reproject(
                    ds_imerg_clean.rio.crs, 
                    shape=(3600, 3600),
                    resampling=Resampling.nearest
                )

                ds_imerg_resampled.to_netcdf(output_file)
                print(f"Resampled IMERG data saved to: {output_file}")
