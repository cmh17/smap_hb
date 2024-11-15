import os
import rioxarray
import geopandas as gpd
import xarray as xr
import numpy as np

"""
Merge, crop, and process the POLARIS data, converting log10-transformed variables back to normal units.
"""

workspace = os.getcwd()

base_input_dir = f"{workspace}/data/polaris"
output_dir = f"{workspace}/data/polaris/processed"

# Define properties and depths
properties = ["silt", "sand","clay", "bd","theta_s", "theta_r","ksat", "ph","om", "lambda", "hb","n", "alpha"]  # List of properties
depths = ['0_5', '5_15', '15_30', '30_60', '60_100', '100_200']

# Load the shapefile for cropping
bbox_path = f"{workspace}/data/huc10/centroid_buffered_bounding_box.shp"
bbox = gpd.read_file(bbox_path)

# Function to merge and load .tif files into xarray
def merge_tif_files(input_folder):
    tif_files = [os.path.join(root, file)
                 for root, dirs, files in os.walk(input_folder)
                 for file in files if file.endswith(".tif")]
    
    if not tif_files:
        print(f"No TIFF files found in {input_folder}")
        return None
    
    datasets = [xr.open_dataarray(tif) for tif in tif_files]
    
    merged_data = datasets[0]  # Use the first dataset as the merged result
    for ds in datasets[1:]:
        merged_data = merged_data.combine_first(ds)
    
    return merged_data

# Function to apply log10 inverse to log-transformed variables
def apply_log10_inverse(data, property_name):
    if property_name in ['ksat', 'alpha','om','hb']:
        # Convert log10-transformed values back to normal using NumPy
        data = np.power(10, data)
        print(f"Converted log10-transformed {property_name} back to normal units.")
    return data

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
    # cropped_dir = f"{property_dir}/cropped"
    
    os.makedirs(merged_dir, exist_ok=True)
    # os.makedirs(cropped_dir, exist_ok=True)

    for depth in depths:
        input_folder = f"{base_input_dir}/{property}/{depth}"
        merged_output_path = f"{merged_dir}/merged_{property}_{depth}.tif"
        # cropped_output_path = f"{cropped_dir}/cropped_{property}_{depth}.tif"

        # Check if merged output already exists
        if os.path.exists(merged_output_path):
            print(f"Skipping merge for {property} at depth {depth}, file already exists.")
            merged_data = xr.open_dataarray(merged_output_path)
        else:
            # Merge the files if not already done
            merged_data = merge_tif_files(input_folder)
            if merged_data is not None:
                # Apply log10 inverse for log-transformed properties
                merged_data = apply_log10_inverse(merged_data, property)
                merged_data.rio.to_raster(merged_output_path)
                print(f"Merged raster saved to {merged_output_path}")
            else:
                continue  # Skip to next depth if merging failed

        # # Check if cropped output already exists
        # if os.path.exists(cropped_output_path):
        #     print(f"Skipping crop for {property} at depth {depth}, file already exists.")
        # else:
        #     # Crop the merged data if not already done
        #     crop_raster(merged_data, cropped_output_path, bbox)
