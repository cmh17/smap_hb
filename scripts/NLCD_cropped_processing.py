import os
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import numpy as np

# Load the NLCD raster
nlcd_file = os.path.join(os.getcwd(), 'data/nlcd/nlcd_2016_buffered.tif')
nlcd_raster = xr.open_dataarray(nlcd_file)
nlcd_raster = nlcd_raster.rio.write_crs("EPSG:4326", inplace=True)

# Load the SMAP data
smap_file = os.path.join(os.getcwd(), 'data/daily/2015-04-01/SMAPHB_SM_2015-04-01.nc')
smap_raster = xr.open_dataset(smap_file)

# Temporarily rename lat/lon to x/y for rioxarray processing
smap_raster_temp = smap_raster.rename({'lon': 'x', 'lat': 'y'})

# Ensure SMAP data has spatial dimensions set for rioxarray
smap_raster_temp = smap_raster_temp.rio.set_spatial_dims(x_dim='x', y_dim='y')

# Assign CRS to the SMAP raster since it's missing from SMAP outputs
smap_raster_temp = smap_raster_temp.rio.write_crs("EPSG:4326", inplace=True)

# Resample NLCD to match the SMAP data's resolution and extent
nlcd_resampled = nlcd_raster.rio.reproject_match(smap_raster_temp)

# Rename x/y to lat/lon after resampling to match SMAP dataset
nlcd_resampled = nlcd_resampled.rename({'x': 'lon', 'y': 'lat'})

# Squeeze out any meaningless "band" dimensions
nlcd_resampled = nlcd_resampled.squeeze('band', drop=True) if 'band' in nlcd_resampled.dims else nlcd_resampled

# Save the resampled NLCD raster with lat/lon coordinates
resampled_nlcd_file = os.path.join(os.getcwd(), 'data/nlcd/resampled_nlcd_2016.tif')
nlcd_resampled.rio.to_raster(resampled_nlcd_file)
print(f"Resampled NLCD raster saved to: {resampled_nlcd_file}")
