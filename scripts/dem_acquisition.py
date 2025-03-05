# use geopandas environment
import os
import geopandas as gpd
import requests
import rasterio
from rasterio.plot import show
from pyproj import Proj, transform

workspace = os.getcwd()

# Input API key
API_Key = "78770cd6f4ea66a61c65096065d94f07"

# Input URL text
url_base = "https://portal.opentopography.org/API/usgsdem"

bbox_path = "%s/data/huc10/centroid_buffered_bounding_box.shp" % workspace
bbox = gpd.read_file(bbox_path)
bounds = bbox.total_bounds

lon1, lat1, lon2, lat2 = bounds[0], bounds[1], bounds[2], bounds[3]

# Create input URL for GET request
def get_url(url_base, datasetName, roi, outputFormat, API_Key):
    return f"{url_base}?datasetName={datasetName}&south={roi[1]}&north={roi[3]}&west={roi[0]}&east={roi[2]}&outputFormat={outputFormat}&API_Key={API_Key}"

payload = get_url(url_base, "USGS30m", (lon1, lat1, lon2, lat2), "GTiff", API_Key)
print(payload)

# Send GET request
response = requests.get(payload, headers={"accept": "application/octet-stream"})

# Check if the request was successful
if response.status_code == 200:
    # Write content to a file
    outDir = os.path.join(workspace, "data", "dem")
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, "USGS_30m_DEM.tif"), "wb") as f:
        f.write(response.content)
else:
    print("Failed to download the file.")


