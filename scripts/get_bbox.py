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

# Find the centroid
centroid_x = selected_watershed.centroid.x.values[0]
centroid_y = selected_watershed.centroid.y.values[0]


# Reporject from UTM to WGS84
original_proj = Proj('epsg:2278')  # South Central Texas
target_proj = Proj('epsg:4326')  # WGS84

# Transform the coordinates
centroid_lat, centroid_lon = transform(original_proj, target_proj, centroid_x, centroid_y)

# # Output the reprojected bounding coordinates
# print(f"Centroid in WGS84: ({centroid_lon}, {centroid_lat})")

# Create a polygon from the bounding box coordinates
bbox = box(centroid_lon-0.5, centroid_lat-0.5, centroid_lon+0.5, centroid_lat+0.5)

# Create a new GeoDataFrame with the bounding box
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")

# Save as a shapefile
output_path = "%s/data/huc10/centroid_bounding_box.shp" % workspace
bbox_gdf.to_file(output_path)