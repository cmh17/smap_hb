import os
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
# import rioxarray as rxr
import numpy as np

import json
from torchvision import transforms

# Allows for selection of time steps
# Upsamples dynamic data and returns dynamic and static data together with same resolution
# Uses standardized data

# Transformer
class ZScore(object):
    """
    Channel‑wise (C×H×W)  standardization  (x‑mean)/std
    """
    def __init__(self, mean, std, eps=1e-7):
        self.mean = torch.tensor(mean)
        self.std  = torch.tensor(std) + eps # prevent division by zero

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean[:, None, None]) / self.std[:, None, None]

class CombinedDataset(Dataset):
    def __init__(
        self,
        root_dir,
        static_stats_file, # json for stats to standardize / scale data
        dynamic_stats_file,
        static_vars=None,
        dynamic_vars=None,
        time_inds=None,
        listoftiles=None
        ):
    
        self.root_dir = root_dir
    
        # If no static vars given, use all
        if static_vars is None:
            sample_static = xr.open_dataset(
                os.path.join(root_dir, (listoftiles or os.listdir(root_dir))[0],
                             "static.nc"))
            static_vars = sorted(sample_static.data_vars)  
            sample_static.close()

        # Same with dynamic
        if dynamic_vars is None:
            sample_dyn = xr.open_dataset(
                os.path.join(root_dir, (listoftiles or os.listdir(root_dir))[0],
                             "dynamic.nc"))
            dynamic_vars = sorted(sample_dyn.data_vars)     
            sample_dyn.close()

        self.static_vars  = static_vars   
        self.dynamic_vars = dynamic_vars

        # If no tiles given, use all tile folders
        if listoftiles is None:
            self.tiles = sorted(os.listdir(root_dir))
        else:
            self.tiles = listoftiles
        
        # If no time indices given, use all
        if time_inds is None:
            with xr.open_dataset(os.path.join(root_dir, self.tiles[0], "dynamic.nc")) as ds:
                time_inds = range(len(ds.time))
        self.samples = [(tile, t) for tile in self.tiles for t in time_inds]

        # Get jsons of statistics to standardize data with
        # Need to use jsons that have stats for the train data being used
        with open(static_stats_file)  as f: s_stats = json.load(f)
        with open(dynamic_stats_file) as f: d_stats = json.load(f)

        means = [s_stats[v]["mean"] for v in self.static_vars] + \
                [d_stats[v]["mean"] for v in self.dynamic_vars]
        stds  = [s_stats[v]["std"]  for v in self.static_vars] + \
                [d_stats[v]["std"]  for v in self.dynamic_vars]

        self.norm = ZScore(means, stds)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tile, t = self.samples[idx]
        tp = os.path.join(self.root_dir, tile)

        static  = self._load_static (os.path.join(tp, "static.nc"))
        dynamic = self._load_dynamic(os.path.join(tp, "dynamic.nc"), t)
        target  = self._load_target (os.path.join(tp, "target.nc"),  t)

        dyn_resampled = np.stack([
            np.full_like(static[0], dynamic[0, 0, 0]),   # IMERG
            np.full_like(static[0], dynamic[1, 0, 0]),   # SMAP
        ])
        predictors = np.concatenate([static, dyn_resampled], axis=0)

        predictors = torch.as_tensor(predictors, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.float32)

        predictors = self.norm(predictors) 

        return predictors, target
    
    def _load_static (self, p):
        return xr.open_dataset(p)[self.static_vars].to_array().values
    
    def _load_dynamic(self, p, t):
        return xr.open_dataset(p).isel(time=t).to_array().values
    
    def _load_target (self, p, t):
        return xr.open_dataset(p).isel(time=t).to_array().values


# def has_bad(t):
#     """
#     Mini-function to return True if there are nans or infs
#     """
#     return torch.isnan(t).any() or torch.isinf(t).any()

# def check_dataset(ds, max_samples=None):
#     loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

#     for idx, (predictor, target) in enumerate(loader):
#         if max_samples is not None and idx >= max_samples:
#             break

#         bad_pred = has_bad(predictor)
#         bad_target = has_bad(target)

#         if bad_pred or bad_target:
#             print(f"\nbad sample #{idx}  ({ds.samples[idx]})")

#             if bad_pred:
#                 bad_idx = torch.isnan(predictor) | torch.isinf(predictor)
#                 print("   predictor bad values:",
#                       predictor[bad_idx][:10].flatten().tolist())
#                 print("   positions (channel, y, x):",
#                       bad_idx.nonzero(as_tuple=False)[:10].tolist())

#             if bad_target:
#                 bad_idx = torch.isnan(target) | torch.isinf(target)
#                 print("   target bad values   :",
#                       target[bad_idx][:10].flatten().tolist())
#                 print("   positions (channel, y, x):",
#                       bad_idx.nonzero(as_tuple=False)[:10].tolist())

#     print("\nScan finished")
    
# def main():
#     workspace = os.path.dirname(os.getcwd())
#     root_dir  = f"{workspace}/data/combined_output/tiles2" # Take in non-standardized tiles
    
#     # Get timesteps from a file
#     ds = xr.open_dataset(os.path.join(workspace, "data", "combined_output", "tiles2", "tile_0_0", "dynamic.nc"))

#     static_vars = ['bd_0_5',
#      'bd_60_100',
#      'clay_0_5',
#      'commercial',
#      'cropland',
#      'elevation',
#      'exurban_high',
#      'exurban_low',
#      'grazing',
#      'hb_0_5',
#      'industrial',
#      'institutional',
#      'ksat_0_5',
#      'mining_barren_land',
#      'n_0_5',
#      'natural_water',
#      'om_0_5',
#      'parks_golf_courses',
#      'pasture',
#      'ph_0_5',
#      'ph_60_100',
#      'recreation_conservation',
#      'reservoirs_canals',
#      'sand_0_5',
#      'silt_0_5',
#      'suburban',
#      'theta_r_0_5',
#      'theta_s_0_5',
#      'timber',
#      'transportation',
#      'urban_high',
#      'urban_low',
#      'wetlands']

#     time_inds = list(range(0,(len(ds.time.values))))

#     years = ds.time.values.astype('datetime64[Y]').astype(int) + 1970

#     time_inds = list(range(list(years).index(2016),list(years).index(2018), 10))

#     train_tiles = []
#     test_tiles = []

#     for i in range(10):
#         for j in range(10):
#             new_tile = f"tile_{i}_{j}"
#             if (i + j) % 2 == 0:
#                 train_tiles.append(new_tile)
#             else:
#                 test_tiles.append(new_tile)

#     training_set = CombinedDataset(root_dir, time_inds=time_inds, listoftiles = train_tiles, static_vars=static_vars,
#                                    static_stats_file=os.path.join(workspace, "data", "combined_output", "static_norm_stats.json"),
#                                    dynamic_stats_file=os.path.join(workspace, "data", "combined_output", "dynamic_norm_stats.json"))

#     check_dataset(training_set)

# if __name__ == "__main__":
#     main()

#     # Check dimensions
#     for predictor, target in training_set:
#         print("predictor shape: ", predictor.shape) # torch.Size([35, 360, 360])
#         print("target shape: ", target.shape) # torch.Size([1, 360, 360])