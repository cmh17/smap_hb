import os
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from nearest_neighbor_2d import fill_na_2d_nearest

def fill_na_2d_for_dataset(ds): # Don't need this since I already interpolated with rio.interpolate_na()
    for var in ds.data_vars:
        da = ds[var]
        if 'lat' in da.dims and 'lon' in da.dims:
            if set(da.dims) == {'lat', 'lon'}:  # static variable
                ds[var] = fill_na_2d_nearest(da)
            else:
                # If there are extra dimensions like time, apply on the lat-lon slice.
                ds[var] = xr.apply_ufunc(
                    fill_na_2d_nearest, 
                    da, 
                    input_core_dims=[["lat", "lon"]],
                    output_core_dims=[["lat", "lon"]],
                    vectorize=True
                )
    return ds

class CombinedDataset(Dataset):
    def __init__(self, root_dir, time_step=-1, listofstatic=None, transform=None):
        self.root_dir = root_dir
        self.tiles = sorted(os.listdir(root_dir))  # List of tile directories
        self.time_step = time_step  # Select time step for dynamic variables
        self.listofstatic = listofstatic if listofstatic is not None else []
        self.transform = transform

    def __len__(self):
        return len(self.tiles)
    
    def load_static_nc(self, path, static_vars=None):
        ds = xr.open_dataset(path)
        # print(path)
        # print(ds)
#         ds = fill_na_2d_for_dataset(ds)
        if static_vars:
            ds = ds[static_vars]
        data = ds.to_array().values  # shape: (C_static, H, W)
        ds.close()
        return data

    def load_dynamic_nc(self, path, time_step):
        ds = xr.open_dataset(path)
        # print(path)
        # print(ds)
#         ds = fill_na_2d_for_dataset(ds)
        # Just one timestep -- modify this later
        ds = ds.isel(time=slice(time_step, time_step + 1))
        data = ds.to_array().values  # shape: (num_dynamic_vars, T, H, W)
        ds.close()
        return data

    def load_target_nc(self, path, time_step):
        ds = xr.open_dataset(path)
        # print(path)
        # print(ds)
#         ds = fill_na_2d_for_dataset(ds)
        # Select one time step
        ds = ds.isel(time=slice(time_step, time_step + 1))
        data = ds.to_array().values  # shape: (C_target, H, W)
        ds.close()
        return data

    def __getitem__(self, idx):
        tile_name = self.tiles[idx]
        tile_path = os.path.join(self.root_dir, tile_name)

        # Paths to NetCDF files
        dynamic_path = os.path.join(tile_path, "dynamic.nc")
        static_path = os.path.join(tile_path, "static.nc")
        target_path = os.path.join(tile_path, "target.nc")

        # Load data using the specific methods
        dynamic_data = self.load_dynamic_nc(dynamic_path, self.time_step)   # (C_dyn, T, H, W)
        static_data = self.load_static_nc(static_path, self.listofstatic)     # (C_static, H, W)
        target_data = self.load_target_nc(target_path, self.time_step)         # (C_target, H, W)

        # Return them separately (they'll be fused in the model)
        static_data = torch.tensor(static_data, dtype=torch.float32)
        dynamic_data = torch.tensor(dynamic_data, dtype=torch.float32)
        target_data = torch.tensor(target_data, dtype=torch.float32)

        if self.transform:
            static_data = self.transform(static_data)
            dynamic_data = self.transform(dynamic_data)
            target_data = self.transform(target_data)

        return static_data, dynamic_data, target_data

# Example usage:
workspace = os.getcwd()
root_dir = f"{workspace}/data/combined_output/tiles2"
static_vars = ["elevation", "silt_0_5", "sand_0_5", "clay_0_5", 
               "urban_low", "urban_high", "commercial", "industrial"]
dataset = CombinedDataset(root_dir, time_step=500, listofstatic=static_vars)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Check one sample
for static_data, dynamic_data, target_data in loader:
    print("Static shape:", static_data.shape)
    print("Dynamic shape:", dynamic_data.shape)
    print("Target shape:", target_data.shape)


Static shape: torch.Size([1, 8, 360, 360])
Dynamic shape: torch.Size([1, 2, 0, 1, 1])
Target shape: torch.Size([1, 1, 0, 360, 360])
