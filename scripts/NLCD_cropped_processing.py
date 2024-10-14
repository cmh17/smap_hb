import os
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import numpy as np

nlcd_file = os.path.join(os.getcwd(), 'data/nlcd/nlcd_2016_cropped.tif')

nlcd_raster = rioxarray.open_rasterio(nlcd_file)

nlcd_raster = nlcd_raster.rio.write_crs("EPSG:4326", inplace=True)

target_shape = (3600, 3600)  # Same number of rows and columns as other data
target_resolution = 30 / (111320 * np.cos(np.deg2rad(30))) # same as IMERG and SMAP 50 km

nlcd_resampled = nlcd_raster.rio.reproject(
    nlcd_raster.rio.crs,
    shape=target_shape,
    resampling=Resampling.nearest  # Use nearest neighbor for categorical data
)

resampled_nlcd_file = os.path.join(os.getcwd(), 'data/nlcd/resampled_nlcd_2016_cropped.tif')
nlcd_resampled.rio.to_raster(resampled_nlcd_file)

print(f"Resampled NLCD raster saved to: {resampled_nlcd_file}")
