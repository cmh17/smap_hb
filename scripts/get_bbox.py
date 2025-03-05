import os
import datetime
import geopandas as gpd
from pyproj import Proj, transform
from shapely.geometry import box

# Set up working directory
workspace = os.getcwd() # current folder

# Load the shapefile
shapefile_path = "%s/data/huc10/HGAC_HUC_10_Watersheds/HGAC_HUC_10_Watersheds.shp" % workspace
gdf = gpd.read_file(shapefile_path)

# Select the watershed by name
selected_watershed = gdf[gdf['NAME'] == 'Little Cypress Creek-Cypress Creek']
watershed_wgs84 = selected_watershed.to_crs(epsg=4326)

# Find the centroid
centroid_x = selected_watershed.centroid.x.values[0]
centroid_y = selected_watershed.centroid.y.values[0]

# Reproject from UTM to WGS84
original_proj = Proj('epsg:2278')  # South Central Texas, from original data
target_proj = Proj('epsg:4326')  # WGS84

# Transform the coordinates
centroid_lat, centroid_lon = transform(original_proj, target_proj, centroid_x, centroid_y)
centroid_lat = round(centroid_lat, 1)
centroid_lon = round(centroid_lon, 1)

# # Output the reprojected bounding coordinates
print(f"Centroid in WGS84: ({centroid_lon}, {centroid_lat})")

# Create a polygon from the bounding box coordinates
bbox = box(centroid_lon-0.5, centroid_lat-0.5, centroid_lon+0.5, centroid_lat+0.5)

# Create a buffered bounding box
buffered_bbox = bbox.buffer(0.1)

# Print results
print(f"Bounding box coordinates in WGS84: ({centroid_lon-0.5}, {centroid_lat-0.5}), ({centroid_lon+0.5}, {centroid_lat+0.5})")
print(f"Buffered bounding box coordinates in WGS84: ({centroid_lon-0.6}, {centroid_lat-0.6}), ({centroid_lon+0.6}, {centroid_lat+0.6})")

# Create a new GeoDataFrame with the bounding box
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
buffered_bbox_gdf = gpd.GeoDataFrame(geometry=[buffered_bbox], crs="EPSG:4326")

# Save as a shapefile
watershed_path = "%s/data/huc10/cypress_creek_watershed.shp" % workspace
output_path = "%s/data/huc10/centroid_bounding_box.shp" % workspace
buffered_output_path = "%s/data/huc10/centroid_buffered_bounding_box.shp" % workspace

watershed_wgs84.to_file(watershed_path)
bbox_gdf.to_file(output_path)
buffered_bbox_gdf.to_file(buffered_output_path)
