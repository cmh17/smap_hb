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

def create_zarr_template(final_path, variable, lats, lons, chunk, 
                        ag_times, compression_level):

    nlon = len(lons)
    nlat = len(lats)
    nt = len(ag_times)

    # Create a uninitialize xarray template
    template = xr.DataArray(da.empty((nt,nlat,nlon), dtype=np.float32, compute=False),
                            coords=[ag_times, lats, lons],
                            dims=["time", "lat", "lon"])
    template = template.to_dataset(name=variable)

    # Define and rechunk data
    template = template.chunk(chunk)

    # Grab chunk ranges
    clats = template.chunks['lat']
    clons = template.chunks['lon']
    ctimes = template.chunks['time']

    # Double chunk in time
    chunk['time'] = int(np.ceil(chunk['time']/10.))
    template = template.chunk(chunk)
    
    # Define attributes
    attrs = dict(projection="EPSG:4326")
    template = template.assign_attrs(attrs)

    # Create zarr template on the disk
    if os.path.exists(final_path):
        shutil.rmtree(final_path)
    zarr_compressor = Blosc(cname="zstd", clevel=compression_level, shuffle=1)
    zarr_encoding = { variable    : {'_FillValue': -9999,
                                    'compressor': zarr_compressor,
                                    'chunks': (chunk['time'],chunk['lat'],chunk['lon'])},
                                    'time': {'dtype': 'i4'},
                    }
    template.to_zarr(final_path, encoding=zarr_encoding, zarr_format=2, compute=False, consolidated=True, mode='w')
    print(datetime.datetime.now(), 'Data template is ready', flush=True)

    template.close()
    del template

    regions = {}

    # define slicing regions
    regions['lon_slice'] = [(i*clons[0],i*clons[0]+clons[i]-1) for i in range(len(clons))]
    regions['lat_slice'] = [(i*clats[0],i*clats[0]+clats[i]-1) for i in range(len(clats))]
    regions['time_slice'] = [(i*ctimes[0],i*ctimes[0]+ctimes[i]-1) for i in range(len(ctimes))]

    # defining regions ranges
    regions['lat_range'] = [(lats[region[0]],lats[region[1]])
                             for region in regions['lat_slice']]
    regions['lon_range'] = [(lons[region[0]],lons[region[1]])
                             for region in regions['lon_slice']]
    regions['time_range'] = [(ag_times[region[0]],ag_times[region[1]])
                             for region in regions['time_slice']]

    return regions

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

    regions_target = create_zarr_template(output_zarr_target, "smaphb_30m", lats_30m, lons_30m, chunk, 
                        ag_times, compression_level)

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

    regions_dynamic = create_zarr_template(output_zarr_dynamic, "smaphb_10km", lats_10km, lons_10km, chunk,
                        ag_times, compression_level)

    for idx, daily_folder in enumerate(daily_folders[1:10]):
        print(f"Processing folder: {daily_folder}")

        target_ds, dynamic_ds = open_and_prepare_daily_dataset(daily_dir, daily_folder)

        time_stamp = np.datetime64(daily_folder)
        time_index = np.searchsorted(ag_times, time_stamp)

        region_slices_target = {
        "time": slice(time_index, time_index+1),
        "lat":  slice(0, len(lats_30m)),   # covers all lat
        "lon":  slice(0, len(lons_30m)),   # covers all lon
        }

        store_path_target = "combined_output/target.zarr"
        ds_zarr_target = xr.open_zarr(store_path_target)

        target_ds.to_zarr(
            store_path_target,
            region=region_slices_target,
            mode="w"
        )

        region_slices_dynamic = {
        "time": slice(time_index, time_index+1),
        "lat":  slice(0, len(lats_10km)),  
        "lon":  slice(0, len(lons_10km)),   
        }

        store_path_dynamic = "combined_output/dynamic.zarr"
        ds_zarr_dynamic = xr.open_zarr(store_path_dynamic)

        dynamic_ds.to_zarr(
            store_path_dynamic,
            region=region_slices_dynamic,
            mode="w"
        )

        
    print(f"\nTarget Zarr store created/updated at: {output_zarr_target}")
    print(f"Dynamic Zarr store created/updated at: {output_zarr_dynamic}")

if __name__ == "__main__":
    main()
