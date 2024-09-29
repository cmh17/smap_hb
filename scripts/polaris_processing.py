import os
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd

workspace = os.getcwd()

base_input_dir = "%s/data/polaris" % workspace
output_dir = "%s/data/polaris/processed" % workspace

properties = ['sand', 'silt', 'clay', 'ksat']
depths = ['0_5', '5_15', '15_30', '30_60', '60_100', '100_200']

# Function to merge .tif files
def merge_tif_files(input_folder, output_path):
    tif_files = [os.path.join(root, file) 
                 for root, dirs, files in os.walk(input_folder) 
                 for file in files if file.endswith(".tif")]

    if not tif_files:
        print(f"No TIFF files found in {input_folder}")
        return None

    tif_list = [rasterio.open(tif) for tif in tif_files]
    
    merged_tif, out_trans = merge(tif_list)
    out_meta = tif_list[0].meta.copy()
    
    out_meta.update({
        "driver": "GTiff",
        "height": merged_tif.shape[1],
        "width": merged_tif.shape[2],
        "transform": out_trans,
        "count": merged_tif.shape[0]
    })
    
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(merged_tif)
    
    print(f"Merged raster saved to {output_path}")
    return output_path

# Function to crop the merged raster to a bounding box
def crop_raster(input_raster_path, output_cropped_path, bbox):
    with rasterio.open(input_raster_path) as src:
        geom = box(*bbox)
        geo_df = gpd.GeoDataFrame({"geometry": [geom]}, crs=src.crs)
        
        out_image, out_transform = mask(src, geo_df.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        with rasterio.open(output_cropped_path, "w", **out_meta) as dest:
            dest.write(out_image)
        
    print(f"Cropped raster saved to {output_cropped_path}")

# Load the shapefile for cropping
bbox_path = "%s/data/huc10/centroid_bounding_box.shp" % workspace
bbox = gpd.read_file(bbox_path)

bbox = bbox.total_bounds

# loop through properties and depths
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

        # merge and crop
        merged_file = merge_tif_files(input_folder, merged_output_path)
        if merged_file:
            crop_raster(merged_file, cropped_output_path, bbox)
