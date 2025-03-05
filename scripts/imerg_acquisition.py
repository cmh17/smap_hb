import os
import geopandas as gpd
import xarray as xr
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import numpy as np  # Import NumPy

workspace = os.getcwd()
output_folder = os.path.join(workspace, 'data/daily/')
os.makedirs(output_folder, exist_ok=True)

# Load bounding box coordinates in WGS84
bbox_path = os.path.join(workspace, "data/huc10/centroid_buffered_bounding_box.shp") 
bbox_gdf = gpd.read_file(bbox_path)
bbox = bbox_gdf.total_bounds

# Base URL for the GPM IMERG dataset
base_url = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/{year}/{month}/3B-DAY.MS.MRG.3IMERG.{year}{month}{day}-S000000-E235959.V07B.nc4"

for folder_name in os.listdir(output_folder):
    try:
        # Construct the full path to the subfolder
        folder_path = os.path.join(output_folder, folder_name)
        
        # Ensure the folder name matches the date format
        date = datetime.strptime(folder_name, "%Y-%m-%d")

        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        url = base_url.format(year=year, month=month, day=day)

        # Define file paths
        file_name = os.path.join(folder_path, f"3B-DAY.MS.MRG.3IMERG.{year}{month}{day}-S000000-E235959.V07B.nc4")
        cropped_file_name = os.path.join(folder_path, f"cropped_3B-DAY.MS.MRG.{year}{month}{day}-S000000-E235959.V07B.nc4")

        # Check if cropped file exists
        if os.path.exists(cropped_file_name):
            print(f"Cropped file for {folder_name} already exists. Skipping...")
            continue
        # Check if raw file exists, else download
        elif os.path.exists(file_name):
            print(f"Raw file for {folder_name} already downloaded. Processing...")
        else:
            # Ensure the subfolder exists
            os.makedirs(folder_path, exist_ok=True)
            
            response = requests.get(url, auth=HTTPBasicAuth('username', 'password'))

            if response.status_code == 200:
                with open(file_name, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded data for {folder_name}")
            else:
                print(f"Failed to download data for {folder_name}: {response.status_code}")
                continue  # Skip to next date if download fails

        # Open the dataset
        ds = xr.open_dataset(file_name, decode_times=True)
        # Select the subset based on bounding box
        subset = ds.sel(lon=slice(bbox[0], bbox[2]), lat=slice(bbox[1], bbox[3]))

        # Iterate through data variables and apply fillna only to numeric types
        for var in subset.data_vars:
            if np.issubdtype(subset[var].dtype, np.number):
                subset[var] = subset[var].fillna(-9999)
            else:
                print(f"Skipping fillna for non-numeric variable: {var}")

        # Save the cropped dataset
        subset.to_netcdf(cropped_file_name)
        print(f"Saved cropped data for {folder_name} to {cropped_file_name}")

    except Exception as e:
        print(f"Error processing folder {folder_name}: {e}")
