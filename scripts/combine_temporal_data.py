import os
from pathlib import Path
import shutil
import os
import sys
import numpy as np
import xarray as xr
from pandas import Series as pd_Series
import datetime
from dateutil.relativedelta import relativedelta
import warnings
import gc
import dask.array as da
import zarr
from numcodecs import Blosc
from itertools import product
import rioxarray

"""
Incrementally combine SMAP 30m, SMAP 10km, and IMERG 10km daily data 
into two separate Zarr stores:
- target.zarr for the SMAP 30m data,
- dynamic.zarr for the SMAP 10km and IMERG 10km data.
"""

def create_zarr_template(
    final_path, 
    var_names, 
    lats, 
    lons, 
    ag_times, 
    chunk, 
    compression_level, 
    additional_attrs=None
):
    """
    Create a lazy Zarr template for the given variables and coordinate arrays,
    without allocating a massive NumPy array. 
    Uses Dask `da.empty` to define the shape lazily.

    :param final_path: Path to save the Zarr store
    :param var_names: List of variable names (e.g. ["smaphb_30m"] or ["smaphb_10km", "imerg_10km"])
    :param lats: 1D array of lat coords (length nlat)
    :param lons: 1D array of lon coords (length nlon)
    :param ag_times: 1D array of time coords (length nt)
    :param chunk: Dictionary specifying chunk sizes, e.g. {"time": 1, "lat": 360, "lon": 360}
    :param compression_level: Integer for Blosc compression level (0..9)
    :param additional_attrs: Optional dictionary of global attributes (e.g. {"projection": "EPSG:4326"})
    """
    # Prepare coordinates
    nt = len(ag_times)
    nlat = len(lats)
    nlon = len(lons)

    # Convert chunk dict into a tuple for Dask
    # E.g. chunk={"time":1,"lat":360,"lon":360} => (1,360,360)
    # If any dimension is missing from chunk, default to full size.
    time_chunk = chunk.get("time", nt)
    lat_chunk = chunk.get("lat", nlat)
    lon_chunk = chunk.get("lon", nlon)

    # Create a Dask array that doesn't store data in memory
    # shape = (nt, nlat, nlon), chunk shape = (time_chunk, lat_chunk, lon_chunk)
    # This array is "empty": meaning uninitialized. We won't physically compute it.
    lazy_array = da.empty(
        (nt, nlat, nlon),
        chunks=(time_chunk, lat_chunk, lon_chunk),
        dtype=np.float32
    )

    # Build an xarray Dataset with each variable referencing the same lazy_array
    # (We just replicate it for each variable, so each variable has the same shape.)
    ds_vars = {}
    for var in var_names:
        ds_vars[var] = (("time", "lat", "lon"), lazy_array)

    ds = xr.Dataset(
        ds_vars,
        coords={"time": ag_times, "lat": lats, "lon": lons}
    )

    # Assign optional global attributes
    if additional_attrs is not None:
        ds = ds.assign_attrs(additional_attrs)

    # Build the encoding dictionary for each variable
    compressor = Blosc(cname="zstd", clevel=compression_level, shuffle=1)
    encoding = {}
    for var in var_names:
        encoding[var] = {
            "_FillValue": -9999,
            "compressor": compressor,
            # We replicate the chunk shape for safety
            "chunks": (time_chunk, lat_chunk, lon_chunk)
        }

    # Also encode the 'time' coordinate as an integer
    encoding["time"] = {"dtype": "i4"}

    # Remove any existing Zarr store
    if os.path.exists(final_path):
        shutil.rmtree(final_path)

    # Write the dataset to Zarr, using mode="w" or mode="w-"
    # compute=False ensures we don't try to fill the array with real data
    ds.to_zarr(
        final_path,
        encoding=encoding,
        zarr_format=2,    # Or omit if you'd rather use zarr v3
        compute=False,
        consolidated=True,
        mode="w"
    )

    print(datetime.datetime.now(), f"Data template is ready for {var_names}")
    return ds


# daily_folders = daily_folders[:10]  # For testing, just use the first 10 days

def open_and_prepare_daily_dataset(daily_dir, daily_folder):
    """
    Open that day's 30m SMAP, 10km SMAP, and 10km IMERG data using Dask,
    rename lat/lon as needed, expand dims to have a single 'time' for the day,
    and merge them into a single Dataset.

    - Initially tried putting in one file, but lots of issues with excessive dimensions
    """
    folder_path = os.path.join(daily_dir, daily_folder)
    time_stamp = np.datetime64(daily_folder)  

    # Prepare lists of DataArrays for each resolution
    target_dataarrays = []   # 30m target data
    dynamic_dataarrays = []  # 10km dynamic data

    # 30 m SMAP-HB (target)
    smap_30m_file = os.path.join(folder_path, f"SMAP-HB_surface-soil-moisture_30m_daily_{daily_folder}.nc")
    da_30m = xr.open_dataarray(smap_30m_file)
    # da_30m = da_30m.rename({"lat": "lat_30m", "lon": "lon_30m"})
    da_30m = da_30m.expand_dims(time=[time_stamp])
    da_30m.name = "smaphb_30m"
    target_dataarrays.append(da_30m)

    # 10 km SMAP-HB (dynamic)
    smap_10km_file = os.path.join(folder_path, f"SMAP-HB_surface-soil-moisture_10000m_daily_{daily_folder}.nc")
    da_10km = xr.open_dataarray(smap_10km_file)
    # da_10km = da_10km.rename({"lat": "lat_10km", "lon": "lon_10km"})
    da_10km = da_10km.expand_dims(time=[time_stamp])
    da_10km.name = "smaphb_10km"
    dynamic_dataarrays.append(da_10km)

    # 10 km IMERG (dynamic)
    time_str = daily_folder.replace("-", "")
    imerg_file = os.path.join(folder_path, f"cropped_3B-DAY.MS.MRG.3IMERG.{time_str}-S000000-E235959.V07B.nc4")
    ds_imerg = xr.open_dataset(imerg_file)
    da_imerg = ds_imerg["precipitation"]
    # da_imerg = da_imerg.rename({"lat": "lat_10km", "lon": "lon_10km"})
    da_imerg = da_imerg.assign_coords(time=[time_stamp]) # Change it to match others' time format
    da_imerg.name = "imerg_10km"

    # Reproject IMERG to match SMAP 10km
    da_imerg = da_imerg.transpose('time', 'lat', 'lon')
    da_10km.rio.set_spatial_dims('lon', 'lat', inplace=True)
    da_imerg.rio.set_spatial_dims('lon', 'lat', inplace=True)
    da_imerg = da_imerg.rio.reproject_match(da_10km)
    da_imerg = da_imerg.rename({"y": "lat", "x": "lon"})
    dynamic_dataarrays.append(da_imerg)

    # Merge each group into separate Datasets
    target_ds = xr.merge(target_dataarrays, compat="override")
    dynamic_ds = xr.merge(dynamic_dataarrays, compat="override")

    # Drop spatial ref
    target_ds = target_ds.drop_vars("spatial_ref")
    dynamic_ds = dynamic_ds.drop_vars("spatial_ref")

    return target_ds, dynamic_ds

def main():
    # === CONFIGURATION === #
    workspace = os.getcwd()
    daily_dir = os.path.join(workspace, "data", "daily")
    output_dir = os.path.join(workspace, "data", "combined_output")
    output_zarr_target = os.path.join(output_dir, "target.zarr")
    output_zarr_dynamic = os.path.join(output_dir, "dynamic.zarr")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Sorting daily folders to ensure chronological order
    daily_folders = sorted([
        d for d in os.listdir(daily_dir)
        if os.path.isdir(os.path.join(daily_dir, d))
    ])

    # Look at first day's data to get spatial information
    first_day_folder = daily_folders[0]
    first_day_path = os.path.join(daily_dir, first_day_folder)
    first_day_file = os.path.join(first_day_path, f"SMAP-HB_surface-soil-moisture_30m_daily_{first_day_folder}.nc")
    first_day_ds = xr.open_dataset(first_day_file)
    lats_30m = first_day_ds["lat"].values
    lons_30m = first_day_ds["lon"].values
    chunk = {"time": 1, "lat": 360, "lon": 360}
    compression_level = 5
    # Create ag_times from daily_folders
    ag_times = np.array([np.datetime64(d) for d in daily_folders])

    regions_target = create_zarr_template(
        final_path=output_zarr_target,
        var_names=["smaphb_30m"],
        lats=lats_30m,
        lons=lons_30m,
        ag_times=ag_times,
        chunk={"time": 1, "lat": 360, "lon": 360},
        compression_level=5,
        additional_attrs={"projection": "EPSG:4326"}
    )

    # Create dynamic.zarr template
    # Look at first day's data to get spatial information
    first_day_folder = daily_folders[0]
    first_day_path = os.path.join(daily_dir, first_day_folder)
    first_day_file = os.path.join(first_day_path, f"SMAP-HB_surface-soil-moisture_10000m_daily_{first_day_folder}.nc")
    first_day_ds = xr.open_dataset(first_day_file)
    lats_10km = first_day_ds["lat"].values
    lons_10km = first_day_ds["lon"].values
    chunk = {"time": 1, "lat": 1, "lon": 1}
    compression_level = 5

    regions_dynamic = create_zarr_template(
        final_path=output_zarr_dynamic,
        var_names=["smaphb_10km", "imerg_10km"],
        lats=lats_10km,
        lons=lons_10km,
        ag_times=ag_times,
        chunk={"time": 1, "lat": 1, "lon": 1},
        compression_level=5,
        additional_attrs={"projection": "EPSG:4326"}
    )

    for idx, daily_folder in enumerate(daily_folders):
        print(f"Processing folder: {daily_folder}")

        target_ds, dynamic_ds = open_and_prepare_daily_dataset(daily_dir, daily_folder)
        
        # print(target_ds)
        # print(dynamic_ds)

        time_stamp = np.datetime64(daily_folder)
        time_index = np.searchsorted(ag_times, time_stamp)

        region_slices_target = {
        "time": slice(time_index, time_index+1),
        "lat":  slice(0, len(lats_30m)), 
        "lon":  slice(0, len(lons_30m)), 
        }

        store_path_target = output_zarr_target
        ds_zarr_target = xr.open_zarr(store_path_target)

        # template_target = xr.open_zarr(output_zarr_target)
        # print("Template lat:", template_target["lat"].values[:5])

        # template_dynamic = xr.open_zarr(output_zarr_dynamic)
        # print("Template lat:", template_dynamic["lat"].values[:5])

        # # For a daily dataset:
        # print("Target lat:", target_ds["lat"].values[:5])
        # print("Dynamic lat:", dynamic_ds["lat"].values[:5])

        target_ds.to_zarr(
            store_path_target,
            region=region_slices_target,
            mode="a"
        )

        # Verify the day was written
        test_ds = xr.open_zarr(store_path_target, consolidated=True)
        today_ds = test_ds.sel(time=time_stamp)
        # print("Min / Max for", daily_folder, ":",
        #     today_ds["smaphb_30m"].min().values,
        #     today_ds["smaphb_30m"].max().values)

        region_slices_dynamic = {
        "time": slice(time_index, time_index+1),
        "lat":  slice(0, len(lats_10km)),  
        "lon":  slice(0, len(lons_10km)),   
        }

        store_path_dynamic = output_zarr_dynamic
        ds_zarr_dynamic = xr.open_zarr(store_path_dynamic)

        dynamic_ds.to_zarr(
            store_path_dynamic,
            region=region_slices_dynamic,
            mode="a"
        )

        test_ds_dyn = xr.open_zarr(store_path_dynamic, consolidated=True)
        today_dyn = test_ds_dyn.sel(time=time_stamp)
        # print("Dynamic Min / Max for", daily_folder, ":",
        #     today_dyn["smaphb_10km"].min().values,
        #     today_dyn["smaphb_10km"].max().values,
        #     today_dyn["imerg_10km"].min().values,
        #     today_dyn["imerg_10km"].max().values)

  
    print(f"\nTarget Zarr store created/updated at: {output_zarr_target}")
    print(f"Dynamic Zarr store created/updated at: {output_zarr_dynamic}")

if __name__ == "__main__":
    main()
