import os
import rioxarray
import geopandas as gpd
import xarray as xr

"""
Merge and crop the POLARIS data and organize them into folders for each property.
"""

workspace = os.path.dirname(os.getcwd())

base_input_dir = f"{workspace}/data/polaris"
output_dir = f"{workspace}/data/polaris/processed"

properties = ['sand', 'silt', 'clay', 'ksat']
depths = ['0_5', '5_15', '15_30', '30_60', '60_100', '100_200']

# Load the shapefile for cropping
bbox_path = f"{workspace}/data/huc10/centroid_bounding_box.shp"
bbox = gpd.read_file(bbox_path)

# Check and reproject bbox to match the CRS of the raster
bbox = bbox.to_crs("EPSG:4326")  # Assuming WGS84 CRS, adjust if needed

# Get the bounding box coordinates
bbox_bounds = bbox.total_bounds  # [minx, miny, maxx, maxy]

# Function to merge and load .tif files into xarray
def merge_tif_files(input_folder):
    tif_files = [os.path.join(root, file)
                 for root, dirs, files in os.walk(input_folder)
                 for file in files if file.endswith(".tif")]
    
    if not tif_files:
        print(f"No TIFF files found in {input_folder}")
        return None
    
    datasets = [rioxarray.open_rasterio(tif) for tif in tif_files]
    
    merged_data = datasets[0]  # Use the first dataset as the merged result
    for ds in datasets[1:]:
        merged_data = merged_data.combine_first(ds)
    
    return merged_data


def crop_raster(input_data, output_cropped_path, bbox):
    if not input_data.rio.crs == "EPSG:4326":
        input_data = input_data.rio.reproject("EPSG:4326")

    # Crop using the bounding box (bbox_bounds = [minx, miny, maxx, maxy])
    cropped_data = input_data.rio.clip(bbox.geometry, bbox.crs, drop=True)

    cropped_data.rio.to_raster(output_cropped_path)
    print(f"Cropped raster saved to {output_cropped_path}")

# Loop through properties and depths
for property in properties:
    property_dir = f"{output_dir}/{property}"
    merged_dir = f"{property_dir}/merged"
    cropped_dir = f"{property_dir}/cropped"
    
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    for depth in depths:
        input_folder = f"{base_input_dir}/{property}/{depth}"
        merged_output_path = f"{merged_dir}/merged_{property}_{depth}.tif"
        cropped_output_path = f"{cropped_dir}/cropped_{property}_{depth}.tif"

        # Merge and crop the files
        merged_data = merge_tif_files(input_folder)
        if merged_data is not None:
            crop_raster(merged_data, cropped_output_path, bbox)
