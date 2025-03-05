import os
import rioxarray
import xarray as xr
import geopandas as gpd
import numpy as np

"""
Merge, crop, reproject DEM, ICLUS, and POLARIS data, and combine into a single NetCDF file to match SMAP data.
"""

workspace = os.getcwd()

static_datasets = []   # 30m static data

output_dir = os.path.join(workspace, "data/combined_output")
dem_file = os.path.join(workspace, "data/dem/usgs_30m_dem.nc")
iclus_file = os.path.join(workspace, "data/iclus/iclus_2020_ssp2_rcp45_one_hot.nc")
polaris_file = os.path.join(workspace, "data/polaris/processed/combined_polaris_data.nc")

# Use a SMAP dataset as the template for reprojection
smap_file = os.path.join(workspace, "data/daily/2015-04-01/SMAP-HB_surface-soil-moisture_30m_daily_2015-04-01.nc")
smap_raster = xr.open_dataarray(smap_file)

# Rename dimensions to 'x' and 'y' for reprojection compatibility and set CRS
smap_raster = smap_raster.rename({'lon': 'x', 'lat': 'y'})
smap_raster = smap_raster.rio.set_spatial_dims(x_dim='x', y_dim='y')
if not smap_raster.rio.crs:
    smap_raster = smap_raster.rio.write_crs("EPSG:4326", inplace=True)

# Load all data
dem_data = xr.open_dataarray(dem_file)
iclus_one_hot = xr.open_dataset(iclus_file)
polaris_data = xr.open_dataset(polaris_file)

# Set spatial dims for other datasets
dem_data = dem_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
iclus_one_hot = iclus_one_hot.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
polaris_data = polaris_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

# Reproject all data to match SMAP resolution
dem_data = dem_data.rio.reproject_match(smap_raster)
iclus_one_hot = iclus_one_hot.rio.reproject_match(smap_raster)
polaris_data = polaris_data.rio.reproject_match(smap_raster)

# Crop all datasets to the target region
crop_bounds = {"x": slice(-96.2, -95.2), "y": slice(29.5, 30.5)}
dem_data = dem_data.sel(**crop_bounds)
iclus_one_hot = iclus_one_hot.sel(**crop_bounds)
polaris_data = polaris_data.sel(**crop_bounds)

# Combine all datasets into a single dictionary
print("Combining all datasets into one file...")

static_datasets.append(dem_data)
static_datasets.append(iclus_one_hot)
static_datasets.append(polaris_data)

static_ds = xr.merge(static_datasets, compat="override")

# Rename x and y to lon and lat
static_ds = static_ds.rename({'x': 'lon', 'y': 'lat'})
static_ds = static_ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')

# Rename band_data to elevation
static_ds = static_ds.rename({"band_data": "elevation"})

# Drop band variable
static_ds = static_ds.drop_vars("band")

target_chunks = {"lat": 360, "lon": 360}
static_ds = static_ds.chunk(target_chunks)

# add metadata for crs
static_ds = static_ds.assign_attrs(projection="EPSG:4326")

# Save the final combined dataset
output_zarr_dir = os.path.join(output_dir, "combined_static_data2.zarr")
static_ds.to_zarr(output_zarr_dir, mode="w")

print(f"Combined dataset saved to: {output_zarr_dir}")

