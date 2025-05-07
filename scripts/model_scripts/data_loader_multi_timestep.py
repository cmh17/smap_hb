import os
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np

class CombinedDataset(Dataset):
    def __init__(self, root_dir, time_inds=None, listoftiles=None, listofstatic=None, transform=None):
        """
        Each sample is (static_data, dynamic_data, target_data) for exactly
        one tile and one time_index.
        
        static_data: (C_static, H, W)
        dynamic_data: (C_dyn, H, W)
        target_data: (C_target, H, W)
        """
        self.root_dir = root_dir
        
        # If no tiles list provided, use all
        if not listoftiles:
            listoftiles = sorted(os.listdir(root_dir))
        self.tiles = listoftiles
        
        # Build a list of (tileName, timeIndex) pairs
        self.samples = []
 
        if time_inds is None:
            sample_tile_path = os.path.join(root_dir, self.tiles[0], "dynamic.nc")
            ds = xr.open_dataset(sample_tile_path)
            time_inds = list(range(len(ds.time)))  # e.g. [0..(numTimes-1)]
            ds.close()
        
        # Create an entry for each (tile, timeIndex)
        for tile_name in self.tiles:
            tile_path = os.path.join(root_dir, tile_name)
            
            for t in time_inds:
                self.samples.append((tile_name, t))
            
            # These will thus be ordered (tile number) + (timestep)
        
        self.listofstatic = listofstatic if listofstatic else []
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # key is tile name and tile index together
        tile_name, time_index = self.samples[idx]
        
        tile_path = os.path.join(self.root_dir, tile_name)
        
        # Paths to NetCDF files
        dynamic_path = os.path.join(tile_path, "dynamic.nc")
        static_path  = os.path.join(tile_path, "static.nc")
        target_path  = os.path.join(tile_path, "target.nc")

        dynamic_data = self.load_dynamic_nc(dynamic_path, time_index)
        
        static_data  = self.load_static_nc(static_path)
        
        target_data  = self.load_target_nc(target_path, time_index)
        
        # Convert to torch
        dynamic_data = torch.tensor(dynamic_data, dtype=torch.float32)
        static_data  = torch.tensor(static_data,  dtype=torch.float32)
        target_data  = torch.tensor(target_data,  dtype=torch.float32)

        if self.transform:
            dynamic_data = self.transform(dynamic_data)
            static_data  = self.transform(static_data)
            target_data  = self.transform(target_data)

        return static_data, dynamic_data, target_data

    def load_static_nc(self, path):
        ds = xr.open_dataset(path)
        if self.listofstatic:
            ds = ds[self.listofstatic]
        arr = ds.to_array().values  # shape (C_static, H, W)
        ds.close()
        return arr

    def load_dynamic_nc(self, path, time_index):
        ds = xr.open_dataset(path)
        # Select single time step => shape (C_dyn, H, W)
        ds_sel = ds.isel(time=time_index)
        arr = ds_sel.to_array().values
        ds.close()
        return arr

    def load_target_nc(self, path, time_index):
        ds = xr.open_dataset(path)
        ds_sel = ds.isel(time=time_index)
        arr = ds_sel.to_array().values  # shape (C_target, H, W)
        ds.close()
        return arr
    
# # Example usage:
# workspace = os.path.dirname(os.getcwd())
# root_dir = f"{workspace}/data/combined_output/tiles2"

# ds = xr.open_dataset(os.path.join(workspace, "data", "combined_output", "tiles2", "tile_0_0", "dynamic.nc"))

# time_inds = list(range(0,(len(ds.time.values))))

# years = ds.time.values.astype('datetime64[Y]').astype(int) + 1970

# time_inds = list(range(list(years).index(2016),list(years).index(2018), 10))

# train_tiles = []
# test_tiles = []

# for i in range(10):
#     for j in range(10):
#         new_tile = f"tile_{i}_{j}"
#         if (i + j) % 2 == 0:
#             train_tiles.append(new_tile)
#         else:
#             test_tiles.append(new_tile)
            
# training_set = CombinedDataset(root_dir, time_inds=time_inds, listoftiles = train_tiles)
# validation_set = CombinedDataset(root_dir, time_inds=time_inds, listoftiles = test_tiles)


# # dataset = CombinedDataset(root_dir, 
# #                           time_inds = [1,3,5,7,9], 
# #                           listoftiles = ["tile_1_3", "tile_4_9"],
# #                           listofstatic=static_vars)

# training_loader = DataLoader(training_set, batch_size=1, shuffle=True)
# validation_loader = DataLoader(validation_set, batch_size=1, shuffle=True)
    
# # Check one sample
# for static_data, dynamic_data, target_data in training_loader:
#     print("Static shape:", static_data.shape)
#     print("Dynamic shape:", dynamic_data.shape)
#     print("Target shape:", target_data.shape)
#     break

# print(loader.dataset.tiles)
# Static shape: torch.Size([1, 8, 360, 360])
# Dynamic shape: torch.Size([1, 2, 0, 1, 1])
# Target shape: torch.Size([1, 1, 0, 360, 360])