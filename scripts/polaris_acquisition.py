import os
import requests
import geopandas as gpd
import math

workspace = os.getcwd()

output_base_dir = "%s/data/polaris" % workspace

# Make a function to build the links
def build_polaris_urls(properties, percentiles, depths, lat_ranges, lon_ranges):
    base_url = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0"
    urls = []

    for prop in properties:
        for perc in percentiles:
            for depth in depths:
                for lat_range in lat_ranges:
                    for lon_range in lon_ranges:
                        lat_str = f"lat{lat_range[0]}{lat_range[1]}"
                        lon_str = f"lon{lon_range[0]}{lon_range[1]}"
                        url = f"{base_url}/{prop}/{perc}/{depth}/{lat_str}_{lon_str}.tif"
                        urls.append(url)

    return urls

# Define parameters
properties = ["sand", "silt", "clay", "ksat"]  # List of properties
percentiles = ["p50"]  # Only using p50, but you can add more
depths = ["0_5", "5_15", "15_30", "30_60", "60_100", "100_200"]  # Depth ranges

# Select degrees needed programmatically
bbox_path = "%s/data/huc10/centroid_bounding_box.shp" % workspace
bbox = gpd.read_file(bbox_path)
bounds = bbox.total_bounds

lon1, lat1, lon2, lat2 = bounds[0], bounds[1], bounds[2], bounds[3]

lat_min = math.floor(lat1)
lat_max = math.ceil(lat2)
lon_min = math.floor(lon1)
lon_max = math.ceil(lon2)

# Create integer ranges for latitude and longitude
lat_ranges = [(i, i+1) for i in range(lat_min, lat_max)]
lon_ranges = [(i, i+1) for i in range(lon_min, lon_max)]

# Generate URLs
urls = build_polaris_urls(properties, percentiles, depths, lat_ranges, lon_ranges)

# Set up folders to organize thee data
for url in urls:
# Determine subdir
    if "sand" in url:
        subdir = "sand"
    elif "silt" in url:
        subdir = "silt"
    elif "clay" in url:
        subdir = "clay"
    elif "ksat" in url:
        subdir = "ksat"
    else:
        subdir = ""  # Put other things in parent

    # Get subsubdir - probably a more efficient way to do this
    if "0_5" in url:
        subsubdir = "0_5"
    elif "5_15" in url:
        subsubdir = "5_15"
    elif "15_30" in url:
        subsubdir = "15_30"
    elif "30_60" in url:
        subsubdir = "30_60"
    elif "60_100" in url:
        subsubdir = "60_100"
    elif "100_200" in url:
        subsubdir = "100_200"
    else:
        subsubdir = ""

    file_name = url.split("/")[-1]

    output_dir = os.path.join(output_base_dir, subdir, subsubdir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, file_name)

    print(f"Downloading {file_name} to {output_dir}...")

    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        # Save the file to disk
        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"{file_name} downloaded successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file_name}: {e}")

print("Download process completed.")

