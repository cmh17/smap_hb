import os
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
# import rioxarray as rxr
import numpy as np

# Allows for selection of time steps
# Upsamples dynamic data and returns dynamic and static data together with same resolution
# Uses fixed SMAPHB 10 km data from npy file... temporary fix
# Still get imerg from

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
        smaphb_10km_path = os.path.join(os.path.dirname(root_dir), "new_smaphb_10km.npy")
        dynamic_path = os.path.join(tile_path, "dynamic.nc")
        static_path  = os.path.join(tile_path, "static.nc")
        target_path  = os.path.join(tile_path, "target.nc")

        dynamic_data = self.load_dynamic_nc(dynamic_path, time_index)
        smaphb_10km = np.load(smaphb_10km_path)
        static_data  = self.load_static_nc(static_path)
        
        # Get smaphb_10km value for this tile
        # Need to make this more robust if we ever use >10x10
        x_index = int(tile_name[5])
        y_index = int(tile_name[7])
        smapbh_10km_value = smaphb_10km[time_index, x_index, y_index]
        
        # Make new dynamic layer using imerg and smaphb_10km
        # Note: in the dynamic file, the 0th var is imerg and the 1st var is smap
        dynamic_resampled = np.stack([np.full([static_data.shape[1],static_data.shape[2]],dynamic_data[0,0,0]),
                                      np.full([static_data.shape[1],static_data.shape[2]],smapbh_10km_value)])
        
        predictor_data = np.concatenate([static_data, dynamic_resampled], axis = 0) # put the 86 and 2 vars together in existing axis
        
        target_data  = self.load_target_nc(target_path, time_index)
        
        # Convert to torch
#         dynamic_data = torch.tensor(dynamic_data, dtype=torch.float32)
#         static_data  = torch.tensor(static_data,  dtype=torch.float32)
        predictor_data = torch.tensor(predictor_data, dtype=torch.float32)
        target_data  = torch.tensor(target_data,  dtype=torch.float32)

        if self.transform:
#             dynamic_data = self.transform(dynamic_data)
#             static_data  = self.transform(static_data)
            predictor_data  = self.transform(predictor_data)
            target_data  = self.transform(target_data)

        return predictor_data, target_data

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

# # # Example usage:
# workspace = os.path.dirname(os.getcwd())
# root_dir = f"{workspace}/data/combined_output/tiles2"

# ds = xr.open_dataset(os.path.join(workspace, "data", "combined_output", "tiles2", "tile_0_0", "dynamic.nc"))

# static_vars = ['bd_0_5',
#  'bd_60_100',
#  'clay_0_5',
#  'commercial',
#  'cropland',
#  'elevation',
#  'exurban_high',
#  'exurban_low',
#  'grazing',
#  'hb_0_5',
#  'industrial',
#  'institutional',
#  'ksat_0_5',
#  'mining_barren_land',
#  'n_0_5',
#  'natural_water',
#  'om_0_5',
#  'parks_golf_courses',
#  'pasture',
#  'ph_0_5',
#  'ph_60_100',
#  'recreation_conservation',
#  'reservoirs_canals',
#  'sand_0_5',
#  'silt_0_5',
#  'suburban',
#  'theta_r_0_5',
#  'theta_s_0_5',
#  'timber',
#  'transportation',
#  'urban_high',
#  'urban_low',
#  'wetlands']


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

# training_set = CombinedDataset(root_dir, time_inds=time_inds, listoftiles = train_tiles, listofstatic=static_vars)
# validation_set = CombinedDataset(root_dir, time_inds=time_inds, listoftiles = test_tiles, listofstatic=static_vars)


# dataset = CombinedDataset(root_dir, 
#                           time_inds = [1,3,5,7,9], 
#                           listoftiles = ["tile_1_3", "tile_4_9"])

# training_loader = DataLoader(training_set, batch_size=1, shuffle=True)
# validation_loader = DataLoader(validation_set, batch_size=1, shuffle=True)

# # Check one sample
# for predictor_data, target_data in training_loader:
#     print("Predictor shape:", predictor_data.shape)
#     print("Target shape:", target_data.shape)
#     break
