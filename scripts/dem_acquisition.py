#!/usr/bin/env python3
from pathlib import Path
import requests
import xarray as xr
import rioxarray
import geopandas as gpd
from pyproj import Proj, transform

def get_url(url_base, datasetName, roi, outputFormat, API_Key):
    """Construct the API URL for the DEM request."""
    return (f"{url_base}?datasetName={datasetName}"
            f"&south={roi[1]}&north={roi[3]}"
            f"&west={roi[0]}&east={roi[2]}"
            f"&outputFormat={outputFormat}&API_Key={API_Key}")

def download_dem(url, out_path):
    """Download the DEM file from the provided URL and save it to out_path."""
    try:
        response = requests.get(url, headers={"accept": "application/octet-stream"})
        response.raise_for_status()
    except Exception as e:
        print("Error downloading DEM:", e)
        raise
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(response.content)
    print(f"DEM successfully saved to: {out_path}")

def main():
    workspace = Path.cwd()

    # Input parameters
    API_Key = "API key here"
    url_base = "https://portal.opentopography.org/API/usgsdem"
    
    # Read bounding box from shapefile
    bbox_path = workspace / "data" / "huc10" / "centroid_buffered_bounding_box.shp"
    if not bbox_path.exists():
        print(f"Shapefile not found: {bbox_path}")
        return

    try:
        bbox = gpd.read_file(bbox_path)
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        return

    bounds = bbox.total_bounds  # [minx, miny, maxx, maxy]
    lon1, lat1, lon2, lat2 = bounds[0], bounds[1], bounds[2], bounds[3]

    # Create the API URL for the DEM download
    payload = get_url(url_base, "USGS30m", (lon1, lat1, lon2, lat2), "GTiff", API_Key)
    print("Request URL:", payload)

    # Download the DEM
    dem_dir = workspace / "data" / "dem"
    dem_file = dem_dir / "USGS_30m_DEM.tif"
    try:
        download_dem(payload, dem_file)
    except Exception as e:
        print("DEM download failed:", e)
        return

    # Load the DEM raster
    try:
        dem_raster = xr.open_dataarray(str(dem_file))
    except Exception as e:
        print("Failed to open DEM file:", e)
        return

    # Ensure the DEM has a CRS
    dem_raster = dem_raster.rio.write_crs("EPSG:4326", inplace=True)

    # Remove the "band" dimension if present
    if "band" in dem_raster.dims:
        dem_raster = dem_raster.isel(band=0, drop=True)

    # Load a SMAP dataset as a template for reprojection
    smap_file = workspace / "data" / "smaphb" / "smaphb_30m" / "SMAP-HB_surface-soil-moisture_30m_daily_netcdf" / "SMAP-HB_surface-soil-moisture_30m_daily_2015-04.nc"
    try:
        smap_raster = xr.open_dataarray(str(smap_file))
    except Exception as e:
        print("Failed to open SMAP file:", e)
        return

    # Rename dimensions to 'x' and 'y' for reprojection compatibility and set CRS
    smap_raster = smap_raster.rename({'lon': 'x', 'lat': 'y'})
    smap_raster = smap_raster.rio.set_spatial_dims(x_dim='x', y_dim='y')
    smap_raster = smap_raster.rio.write_crs("EPSG:4326", inplace=True)

    # Reproject DEM to match SMAP data
    dem_reprojected = dem_raster.rio.reproject_match(smap_raster)

    # Update encoding to use -9999
    if "_FillValue" in dem_reprojected.encoding:
        del dem_reprojected.encoding["_FillValue"]
    dem_reprojected.encoding["_FillValue"] = -9999

    dem_reprojected.name = "elevation"

    # Rename dimensions back to 'lon' and 'lat'
    dem_reprojected = dem_reprojected.rename({'x': 'lon', 'y': 'lat'})
    dem_reprojected = dem_reprojected.rio.set_spatial_dims(x_dim='lon', y_dim='lat')

    dem_reprojected = dem_reprojected.rio.write_crs("EPSG:4326", inplace=True)
    
    # Save the reprojected DEM as a NetCDF file
    reprojected_dem_file = workspace / "data" / "dem" / "usgs_30m_dem.nc"
    try:
        # remove if file already exists
        if reprojected_dem_file.exists():
            reprojected_dem_file.unlink()
        dem_reprojected.to_netcdf(str(reprojected_dem_file))
        print(f"Reprojected DEM raster saved to: {reprojected_dem_file}")
    except Exception as e:
        print("Error saving reprojected DEM:", e)

if __name__ == "__main__":
    main()