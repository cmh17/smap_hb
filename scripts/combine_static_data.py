import os
import rioxarray
import xarray as xr
import geopandas as gpd

"""
Merge, crop, reproject DEM, ICLUS, and POLARIS data, and combine into a single NetCDF file to match SMAP data.
"""

workspace = os.getcwd()

base_input_dir = f"{workspace}/data/polaris"
output_dir = f"{workspace}/data/combined_output"
dem_file = os.path.join(workspace, 'data/dem/USGS_30m_DEM.tif')
iclus_file = os.path.join(workspace, 'data/iclus/ICLUS_v2_1_1_land_use_conus_ssp2_rcp45_hadgem2_es/ICLUS_v2_1_1_land_use_conus_2020_ssp2_rcp45_hadgem2_es.tif')

# Properties and depths for POLARIS
properties = ['sand', 'silt', 'clay', 'ksat', 'theta_s', 'lambda', 'hb', 'alpha']
depths = ['0_5', '5_15', '15_30', '30_60', '60_100', '100_200']

# Load one of the SMAP datasets as a template for reprojection
smap_file = os.path.join(workspace, 'data/daily/2015-04-01/SMAPHB_SM_2015-04-01.nc')
smap_raster = xr.open_dataarray(smap_file)

# Change names of lat and lon so .reproject_match works
smap_raster = smap_raster.rename({'lon': 'x', 'lat': 'y'})  # Ensure spatial dimensions
smap_raster = smap_raster.rio.set_spatial_dims(x_dim='x', y_dim='y')

# Function to reproject rasters to match SMAP data
def reproject_raster(input_data, reference_raster):
    reprojected_data = input_data.rio.reproject_match(reference_raster)
    # Squeeze out any meaningless "band" dimensions
    reprojected_data = reprojected_data.squeeze('band', drop=True) if 'band' in reprojected_data.dims else reprojected_data
    return reprojected_data

# Function to merge and load .tif files into xarray
def merge_tif_files(input_folder):
    tif_files = [os.path.join(root, file) for root, dirs, files in os.walk(input_folder) for file in files if file.endswith(".tif")]
    if not tif_files:
        print(f"No TIFF files found in {input_folder}")
        return None
    datasets = [xr.open_dataarray(tif) for tif in tif_files]
    merged_data = datasets[0]
    for ds in datasets[1:]:
        merged_data = merged_data.combine_first(ds)
    return merged_data

# Load and reproject DEM and ICLUS data
dem_data = xr.open_dataarray(dem_file)
iclus_full = xr.open_dataarray(iclus_file)

# Load gdf to crop ICLUS data
buffered_study_area = gpd.read_file(f'{workspace}/data/huc10/centroid_buffered_bounding_box.shp')

# Clip ICLUS data so it isn't too large to reproject (will take ~9 minutes)
iclus_clipped = iclus_full.rio.clip(buffered_study_area.geometry.values, crs=buffered_study_area.crs)

dem_reprojected = reproject_raster(dem_data, smap_raster)
iclus_reprojected = reproject_raster(iclus_clipped, smap_raster)

# Create a dictionary to store all variables for combination into a single Dataset
combined_data = {
    'DEM': dem_reprojected,
    'ICLUS': iclus_reprojected
}

# Process and add POLARIS data
for property in properties:
    print(f"Processing {property} rasters...")
    for depth in depths:
        print(f"Processing {depth} cm depth files...")
        input_folder = f"{base_input_dir}/{property}/{depth}"
        merged_data = merge_tif_files(input_folder)
        if merged_data is None:
            continue  # Skip to next depth if merging failed
        # Reproject the merged data to match the SMAP data projection and extent
        reprojected_data = reproject_raster(merged_data, smap_raster)
        # Add the reprojected data to the dictionary for combination
        variable_name = f"{property}_{depth}"
        combined_data[variable_name] = reprojected_data

# Combine all the data into a single Dataset
print("Combining all datasets into one .nc...")
combined_dataset = xr.Dataset(combined_data)

combined_dataset = combined_dataset.rename({'x': 'lon', 'y': 'lat'})  # Go back to lat and lon since those are SMAP's labels
combined_dataset = combined_dataset.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
combined_dataset = combined_dataset.fillna(-9999)
combined_dataset = combined_dataset.rio.write_crs("EPSG:4326")

# Save the combined dataset as a NetCDF file
os.makedirs(output_dir, exist_ok=True)
output_nc_file = os.path.join(output_dir, 'combined_dem_iclus_polaris_data.nc')
combined_dataset.to_netcdf(output_nc_file)

print(f"Combined DEM, ICLUS, and POLARIS data saved to {output_nc_file}")
