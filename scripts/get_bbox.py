#!/usr/bin/env python3
from pathlib import Path
import geopandas as gpd
from pyproj import Proj, transform
from shapely.geometry import box
import numpy as np

# Download HUC10 watersheds shapefile from HGAC: https://gishub-h-gac.hub.arcgis.com/datasets/H-GAC::hgac-huc-10-watersheds/explore?location=29.641555%2C-95.500150%2C6.85

def get_selected_watershed(shapefile_path, watershed_name):
    """Read the shapefile and return the selected watershed by name."""
    gdf = gpd.read_file(shapefile_path)
    selected = gdf[gdf['NAME'] == watershed_name]
    if selected.empty:
        raise ValueError(f"No watershed found with name: {watershed_name}")
    return selected

def compute_centroid(selected_watershed, original_epsg='2278', target_epsg='4326'):
    """
    Compute the centroid of the selected watershed and transform it 
    from the original EPSG to the target EPSG. Returns the (latitude, longitude)
    rounded to one decimal place.
    """
    centroid_x = selected_watershed.centroid.x.values[0]
    centroid_y = selected_watershed.centroid.y.values[0]
    
    original_proj = Proj(f"epsg:{original_epsg}")
    target_proj = Proj(f"epsg:{target_epsg}")
    
    centroid_lat, centroid_lon = transform(original_proj, target_proj, centroid_x, centroid_y)
    return round(centroid_lat, 1), round(centroid_lon, 1)

def create_bounding_boxes(centroid_lat, centroid_lon, delta=0.5, buffer_val=0.6):
    """
    Create a bounding box and a buffered bounding box around the centroid.
    Returns two shapely box geometries.
    """
    bbox_coords = np.around([centroid_lon - delta, centroid_lat - delta,
                             centroid_lon + delta, centroid_lat + delta], decimals=1)
    buffered_coords = np.around([centroid_lon - buffer_val, centroid_lat - buffer_val,
                                 centroid_lon + buffer_val, centroid_lat + buffer_val], decimals=1)
    
    bbox = box(*bbox_coords)
    buffered_bbox = box(*buffered_coords)
    
    return bbox, buffered_bbox

def save_bounding_boxes(bbox, buffered_bbox, output_dir):
    """
    Save the bounding box and the buffered bounding box as shapefiles
    in the specified output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
    buffered_bbox_gdf = gpd.GeoDataFrame(geometry=[buffered_bbox], crs="EPSG:4326")
    
    bbox_path = output_dir / "centroid_bounding_box.shp"
    buffered_bbox_path = output_dir / "centroid_buffered_bounding_box.shp"
    
    bbox_gdf.to_file(bbox_path)
    buffered_bbox_gdf.to_file(buffered_bbox_path)
    
    print(f"Saved bounding box shapefile to: {bbox_path}")
    print(f"Saved buffered bounding box shapefile to: {buffered_bbox_path}")

def main():
    # Define workspace using pathlib
    workspace = Path.cwd()
    
    # Define the shapefile path and watershed name
    shapefile_path = workspace / "data" / "huc10" / "HGAC_HUC_10_Watersheds" / "HGAC_HUC_10_Watersheds.shp"
    watershed_name = "Little Cypress Creek-Cypress Creek"
    
    # Select the watershed and (optionally) convert CRS if needed
    selected_watershed = get_selected_watershed(shapefile_path, watershed_name)
    watershed_wgs84 = selected_watershed.to_crs(epsg=4326)
    
    # Compute the centroid in WGS84
    centroid_lat, centroid_lon = compute_centroid(selected_watershed)
    print(f"Centroid coordinates: ({centroid_lat}, {centroid_lon})")
    
    # Create bounding boxes around the centroid
    bbox, buffered_bbox = create_bounding_boxes(centroid_lat, centroid_lon)
    print(f"Bounding box: {bbox}")
    print(f"Buffered bounding box: {buffered_bbox}")
    
    # Save the bounding boxes as shapefiles
    output_dir = workspace / "data" / "huc10"
    save_bounding_boxes(bbox, buffered_bbox, output_dir)

if __name__ == "__main__":
    main()
