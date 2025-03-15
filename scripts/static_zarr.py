import os
import xarray as xr
import numpy as np
import rioxarray
import shutil
import dask.array as da
import numpy as np
import xarray as xr
from numcodecs import Blosc

def create_static_zarr_template(
    final_path,
    var_names,
    lats,
    lons,
    chunk,
    compression_level,
    additional_attrs=None
):
    """
    Create a lazy Zarr template for static data (no time dimension).
    This store will have shape (lat, lon) and chunking given by `chunk`,
    e.g. {'lat': 360, 'lon': 360}.
    """
    nlat = len(lats)
    nlon = len(lons)
    
    # Convert chunk dict -> tuple for Dask
    lat_chunk = chunk.get("lat", nlat)
    lon_chunk = chunk.get("lon", nlon)
    
    # Create a Dask array, shape (nlat, nlon), chunked (lat_chunk, lon_chunk)
    lazy_array = da.empty((nlat, nlon),
                          chunks=(lat_chunk, lon_chunk),
                          dtype=np.float32)
    
    # Build an xarray Dataset with each variable referencing lazy_array
    ds_vars = {}
    for var in var_names:
        ds_vars[var] = (("lat", "lon"), lazy_array)

    ds = xr.Dataset(
        ds_vars,
        coords={"lat": lats, "lon": lons}
    )
    # Add any global attributes you want
    if additional_attrs is not None:
        ds = ds.assign_attrs(additional_attrs)

    # Prepare the Zarr encoding
    compressor = Blosc(cname="zstd", clevel=compression_level, shuffle=1)
    encoding = {}
    attrs = {}
    for var in var_names:
        encoding[var] = {
            # "_FillValue": -9999,
            "compressor": compressor,
            "chunks": (lat_chunk, lon_chunk),
        }

    # Remove existing store if present
    if os.path.exists(final_path):
        shutil.rmtree(final_path)

    # Write the template to Zarr, no actual data is computed
    ds.to_zarr(
        final_path,
        encoding=encoding,
        zarr_format=2,
        compute=False,
        consolidated=True,
        mode="w"
    )
    print("Static template created at:", final_path)
    return ds

def main():
    workspace = os.getcwd()
    
    output_dir = os.path.join(workspace, "data", "combined_output")
    os.makedirs(output_dir, exist_ok=True)
    output_zarr_dir = os.path.join(output_dir, "static.zarr")

    dem_file = os.path.join(workspace, "data", "dem", "usgs_30m_dem.nc")
    iclus_file = os.path.join(workspace, "data", "iclus", "iclus_2020_ssp2_rcp45_one_hot.nc")
    polaris_file = os.path.join(workspace, "data", "polaris", "processed", "combined_polaris_data.nc")

    # Use SMAP as reference for reprojection
    smap_file = os.path.join(
        workspace, "data", "daily", "2015-04-01",
        "SMAP-HB_surface-soil-moisture_30m_daily_2015-04-01.nc"
    )
    smap_raster = xr.open_dataarray(smap_file)

    # Rename dims for reprojection
    smap_raster = smap_raster.rename({"lon": "x", "lat": "y"})
    smap_raster = smap_raster.rio.set_spatial_dims(x_dim="x", y_dim="y")
    if not smap_raster.rio.crs:
        smap_raster = smap_raster.rio.write_crs("EPSG:4326", inplace=True)

    # Load data
    dem_data = xr.open_dataset(dem_file)
    iclus_one_hot = xr.open_dataset(iclus_file)
    polaris_data = xr.open_dataset(polaris_file)

    print(dem_data)
    print(iclus_one_hot)
    print(polaris_data)

    # Set spatial dims
    dem_data = dem_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    iclus_one_hot = iclus_one_hot.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    polaris_data = polaris_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    # Set the dem crs
    if not dem_data.rio.crs:
        dem_data = dem_data.rio.write_crs("EPSG:4326", inplace=True) # sigh

    if "band" in dem_data:
        dem_data = dem_data.drop_vars("band")

    for var_name in polaris_data.data_vars:
        if "band" in polaris_data[var_name].dims:
            polaris_data[var_name] = polaris_data[var_name].squeeze("band", drop=True)

    # Reproject
    dem_data = dem_data.rio.reproject_match(smap_raster)
    iclus_one_hot = iclus_one_hot.rio.reproject_match(smap_raster)
    polaris_data = polaris_data.rio.reproject_match(smap_raster)

    # Crop
    crop_bounds = {"x": slice(-96.2, -95.2), "y": slice(29.5, 30.5)}
    dem_data = dem_data.sel(**crop_bounds)
    iclus_one_hot = iclus_one_hot.sel(**crop_bounds)
    polaris_data = polaris_data.sel(**crop_bounds)

    # Merge everything
    static_datasets = []
    static_datasets.append(dem_data)
    static_datasets.append(iclus_one_hot)
    static_datasets.append(polaris_data)
    static_ds = xr.merge(static_datasets, compat="override")

    # Drop the spatial ref coordinate
    if "spatial_ref" in static_ds:
        static_ds = static_ds.drop_vars("spatial_ref")

    # Rename x,y -> lon,lat
    static_ds = static_ds.rename({"x": "lon", "y": "lat"})
    static_ds = static_ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    # Let's get final lat/lon arrays
    lats = static_ds["lat"].values
    lons = static_ds["lon"].values

    # Include all vars for now
    var_names = list(static_ds.data_vars)

    chunk = {"lat": 360, "lon": 360}
    compression_level = 5

    create_static_zarr_template(
        final_path=output_zarr_dir,
        var_names=var_names,
        lats=lats,
        lons=lons,
        chunk=chunk,
        compression_level=compression_level,
        additional_attrs={"projection": "EPSG:4326"}
    )

    # 2) Write the final data into the template using region slicing
    region_slices = {
        "lat": slice(0, len(lats)),
        "lon": slice(0, len(lons))
    }

    # open the template store
    ds_zarr = xr.open_zarr(output_zarr_dir)

    # remove spatial_ref from the template
    if "spatial_ref" in ds_zarr:
        ds_zarr = ds_zarr.drop_vars("spatial_ref")

    static_ds.attrs["projection"] = "EPSG:4326"

    for var in static_ds.data_vars:
        # Remove any existing _FillValue from attrs
        if "_FillValue" in static_ds[var].attrs:
            del static_ds[var].attrs["_FillValue"]
        # Remove any existing _FillValue in encoding
        if "_FillValue" in static_ds[var].encoding:
            del static_ds[var].encoding["_FillValue"]
        # Now set the _FillValue in encoding
        static_ds[var].encoding["_FillValue"] = -9999

    static_ds.to_zarr(
        output_zarr_dir,
        region=region_slices,
        mode="a"
    )

    print(f"Static data combined and written to: {output_zarr_dir}")

if __name__ == "__main__":
    main()
