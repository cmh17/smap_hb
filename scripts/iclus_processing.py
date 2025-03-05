#!/usr/bin/env python3
from pathlib import Path
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import numpy as np
import requests
import zipfile
import io

def download_and_extract(url, extract_to):
    """
    Download a zip file from the provided URL and extract its contents to the specified directory.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        print("Download successful.")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(path=extract_to)
        print(f"Extracted data to: {extract_to}")
    except Exception as e:
        print("Error downloading and extracting:", e)
        raise

def load_iclus_data(iclus_path):
    """Load ICLUS data from a TIFF file."""
    try:
        iclus = xr.open_dataarray(str(iclus_path))
        print("Loaded ICLUS data.")
        return iclus
    except Exception as e:
        print("Error loading ICLUS data:", e)
        raise

def load_smap_template(smap_file):
    """Load SMAP data as a template for reprojection."""
    try:
        smap_raster = xr.open_dataarray(str(smap_file))
        # Rename spatial dimensions to 'x' and 'y' for reprojection
        smap_raster = smap_raster.rename({'lon': 'x', 'lat': 'y'})
        smap_raster = smap_raster.rio.set_spatial_dims(x_dim='x', y_dim='y')
        smap_raster = smap_raster.rio.write_crs("EPSG:4326", inplace=True)
        print("Loaded and prepared SMAP template.")
        return smap_raster
    except Exception as e:
        print("Error loading SMAP data:", e)
        raise

def clip_data(data, clip_shapefile):
    """Clip data using the provided shapefile geometry."""
    try:
        clipped = data.rio.clip(clip_shapefile.geometry.values, crs=clip_shapefile.crs)
        print("Data clipping successful.")
        return clipped
    except Exception as e:
        print("Clipping failed:", e)
        raise

def one_hot_encode(data, land_cover_dict):
    """Perform one-hot encoding for each land cover type."""
    one_hot_layers = {}
    for level, name in land_cover_dict.items():
        one_hot_layers[name] = (data == level).astype(np.uint8)
    one_hot_ds = xr.Dataset(one_hot_layers)
    # Copy coordinates and attributes from the original DataArray
    one_hot_ds = one_hot_ds.assign_coords(data.coords).assign_attrs(data.attrs)
    # Rename dimensions back to 'lon' and 'lat'
    one_hot_ds = one_hot_ds.rename({'x': 'lon', 'y': 'lat'})
    one_hot_ds = one_hot_ds.fillna(-9999)
    print("One-hot encoding completed.")
    return one_hot_ds

def main():
    workspace = Path.cwd()
    
    # Define file paths
    iclus_dir = workspace / "data" / "iclus"
    iclus_zip_url = "https://gaftp.epa.gov/epadatacommons/ORD/NCEA/ICLUS_v2.1.1/land_use/ICLUS_v2_1_1_land_use_conus_ssp2_rcp45_hadgem2_es.zip"
    # The expected TIFF file path after extraction:
    iclus_path = iclus_dir / "ICLUS_v2_1_1_land_use_conus_ssp2_rcp45_hadgem2_es" / "ICLUS_v2_1_1_land_use_conus_2020_ssp2_rcp45_hadgem2_es.tif"
    
    # Download and extract ICLUS data if it doesn't already exist
    if not iclus_path.exists():
        print("ICLUS data not found. Downloading...")
        download_and_extract(iclus_zip_url, iclus_dir)
    else:
        print("ICLUS data already exists. Skipping download.")
    
    # Load ICLUS data and SMAP template
    iclus = load_iclus_data(iclus_path)
    smap_file = workspace / "data" / "smaphb" / "smaphb_30m" / "SMAP-HB_surface-soil-moisture_30m_daily_netcdf" / "SMAP-HB_surface-soil-moisture_30m_daily_2015-04.nc"
    smap_raster = load_smap_template(smap_file)
    
    # Load the buffered study area shapefile
    clip_shapefile_path = workspace / "data" / "huc10" / "centroid_buffered_bounding_box.shp"
    try:
        buffered_study_area = gpd.read_file(str(clip_shapefile_path))
        print("Loaded buffered study area shapefile.")
    except Exception as e:
        print("Error loading buffered study area shapefile:", e)
        return

    # Clip ICLUS data to the buffered study area
    try:
        iclus_clipped = clip_data(iclus, buffered_study_area)
    except Exception as e:
        print("Clipping ICLUS data failed:", e)
        return
    
    # Set NaNs to -9999
    iclus_clipped = iclus_clipped.fillna(-9999)
    
    # Reproject clipped ICLUS data to match SMAP template
    try:
        iclus_reprojected = iclus_clipped.rio.reproject_match(smap_raster)
        print("Reprojection completed.")
    except Exception as e:
        print("Reprojection failed:", e)
        return
    
    # Save the clipped and reprojected datasets
    clipped_output = iclus_dir / "iclus_2020_ssp2_rcp45_clipped.nc"
    reprojected_output = iclus_dir / "iclus_2020_ssp2_rcp45_reprojected.nc"
    
    iclus_clipped.to_netcdf(str(clipped_output))
    print(f"Clipped ICLUS data saved to: {clipped_output}")
    
    iclus_reprojected.to_netcdf(str(reprojected_output))
    print(f"Reprojected ICLUS data saved to: {reprojected_output}")
    
    # Define land cover mapping dictionary
    land_cover_dict = {
        0: 'natural_water',
        1: 'reservoirs_canals',
        2: 'wetlands',
        3: 'recreation_conservation',
        4: 'timber',
        5: 'grazing',
        6: 'pasture',
        7: 'cropland',
        8: 'mining_barren_land',
        9: 'parks_golf_courses',
        10: 'exurban_low',
        11: 'exurban_high',
        12: 'suburban',
        13: 'urban_low',
        14: 'urban_high',
        15: 'commercial',
        16: 'industrial',
        17: 'institutional',
        18: 'transportation',
    }
    
    # One-hot encode the reprojected ICLUS data
    one_hot_ds = one_hot_encode(iclus_reprojected, land_cover_dict)
    
    one_hot_output = iclus_dir / "iclus_2020_ssp2_rcp45_one_hot.nc"
    one_hot_ds.to_netcdf(str(one_hot_output))
    print(f"One-hot encoded ICLUS data saved to: {one_hot_output}")

if __name__ == "__main__":
    main()
