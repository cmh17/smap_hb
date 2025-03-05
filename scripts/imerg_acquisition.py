#!/usr/bin/env python3
from pathlib import Path
import geopandas as gpd
import xarray as xr
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import numpy as np

def download_data(url, file_path, username, password):
    """Download a file from the URL using HTTP Basic Authentication and save it to file_path."""
    try:
        response = requests.get(url, auth=HTTPBasicAuth(username, password))
        if response.status_code == 200:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(response.content)
            print(f"Downloaded data to: {file_path}")
            return True
        else:
            print(f"Failed to download data from {url}: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error during download from {url}: {e}")
        return False

def process_dataset(file_path, bbox):
    """
    Open a dataset, subset it by the bounding box (bbox),
    and fill NaNs with -9999 for numeric variables.
    """
    try:
        ds = xr.open_dataset(str(file_path), decode_times=True, engine="netcdf4")
    except Exception as e:
        print(f"Error opening dataset {file_path}: {e}")
        return None

    # Set CRS
    ds = ds.rio.write_crs("EPSG:4326")

    # Subset the dataset based on bounding box: [minlon, minlat, maxlon, maxlat]
    subset = ds.sel(lon=slice(bbox[0], bbox[2]), lat=slice(bbox[1], bbox[3]))

    # Drop time bands
    subset = subset.drop_vars(["time_bnds"])
    
    # Iterate through data variables and fill NaNs for numeric types
    for var in subset.data_vars:
        if np.issubdtype(subset[var].dtype, np.number):
            subset[var] = subset[var].fillna(-9999)
        else:
            print(f"Skipping fillna for non-numeric variable: {var}")
    return subset

def main():
    workspace = Path.cwd()
    output_folder = workspace / "data" / "daily"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load bounding box coordinates from the shapefile (WGS84)
    bbox_path = workspace / "data" / "huc10" / "centroid_bounding_box.shp"
    try:
        bbox_gdf = gpd.read_file(str(bbox_path))
        bbox = bbox_gdf.total_bounds  # [minlon, minlat, maxlon, maxlat]
        print(f"Loaded bounding box: {bbox}")
    except Exception as e:
        print(f"Error loading bounding box shapefile: {e}")
        return

    # Base URL for the GPM IMERG dataset
    base_url = ("https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/"
                "{year}/{month}/3B-DAY.MS.MRG.3IMERG.{year}{month}{day}-S000000-E235959.V07B.nc4")

    username = "username"  # Replace with your username
    password = "password"  # Replace with your password

    # Iterate through each subfolder in the output folder (assumed to be dates in YYYY-MM-DD format)
    for folder in output_folder.iterdir():
        print(folder)
        if folder.is_dir():
            try:
                # Ensure folder name matches the date format
                date = datetime.strptime(folder.name, "%Y-%m-%d")
                year = date.strftime("%Y")
                month = date.strftime("%m")
                day = date.strftime("%d")

                # Construct the URL and file paths
                url = base_url.format(year=year, month=month, day=day)
                raw_file = folder / f"3B-DAY.MS.MRG.3IMERG.{year}{month}{day}-S000000-E235959.V07B.nc4"
                cropped_file = folder / f"cropped_3B-DAY.MS.MRG.3IMERG.{year}{month}{day}-S000000-E235959.V07B.nc4"

                # # Skip if cropped file already exists
                # if cropped_file.exists():
                #     print(f"Cropped file for {folder.name} already exists. Skipping...")
                #     continue

                # Download raw file if it doesn't exist
                if not raw_file.exists():
                    print(f"Raw file for {folder.name} not found. Downloading...")
                    if not download_data(url, raw_file, username, password):
                        continue  # Skip if download fails
                else:
                    print(f"Raw file for {folder.name} already exists. Processing...")

                # Process the dataset: subset and fill NaNs
                subset = process_dataset(raw_file, bbox)
                if subset is None:
                    print(f"Failed to process dataset for {folder.name}")
                    continue

                # Save the cropped dataset
                subset.to_netcdf(str(cropped_file), mode="w")
                print(f"Saved cropped data for {folder.name} to {cropped_file}")
            except Exception as e:
                print(f"Error processing folder {folder.name}: {e}")

if __name__ == "__main__":
    main()
