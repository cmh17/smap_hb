import os, json, torch, xarray as xr, numpy as np
from torch.utils.data import Dataset


class ZScore:
    def __init__(self, mean, std, eps=1e-7):
        self.mean = torch.as_tensor(mean)
        self.std  = torch.as_tensor(std) + eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        view = ( -1, ) + (1,)*(x.ndim-1)          # (C,1,1)  or  (C,)
        return (x - self.mean.view(*view)) / self.std.view(*view)

class CombinedDataset(Dataset):
    """
    returns
        static_tensor: (C_static, 360, 360)  – 33 channels (for now), z-scored
        dyn_vector: (2,)                  –   IMERG, SMAP, z-scored
        target_tensor: (1, 360, 360)
    """
    def __init__(self,
                 root_dir: str,
                 static_stats_file: str,
                 dynamic_stats_file: str,
                 static_vars = None,
                 dynamic_vars = None,
                 time_inds = None,
                 listoftiles= None):

        self.root_dir = root_dir

        # Var names
        if static_vars is None:
            with xr.open_dataset(os.path.join(root_dir,
                                              (listoftiles or os.listdir(root_dir))[0],
                                              "static.nc")) as ds:
                static_vars = list(ds.data_vars)
        if dynamic_vars is None:
            with xr.open_dataset(os.path.join(root_dir,
                                              (listoftiles or os.listdir(root_dir))[0],
                                              "dynamic.nc")) as ds:
                dynamic_vars = list(ds.data_vars)

        self.static_vars, self.dynamic_vars = static_vars, dynamic_vars

        # Tile timestamp list
        self.tiles = listoftiles or sorted(os.listdir(root_dir))
        if time_inds is None:
            with xr.open_dataset(os.path.join(root_dir, self.tiles[0],
                                              "dynamic.nc")) as ds:
                time_inds = range(len(ds.time))
        self.samples = [(tile, t) for tile in self.tiles for t in time_inds]

        # Get norm stats for static and dynamic separately
        with open(static_stats_file)  as f: s_stats = json.load(f)
        with open(dynamic_stats_file) as f: d_stats = json.load(f)

        # Use helper function to normalize w dataset-specific stats
        self.norm_static = ZScore([s_stats[v]['mean'] for v in self.static_vars],
                                  [s_stats[v]['std']  for v in self.static_vars])

        self.norm_dyn    = ZScore([d_stats[v]['mean'] for v in self.dynamic_vars],
                                  [d_stats[v]['std']  for v in self.dynamic_vars])


    def __len__(self): return len(self.samples)
    
    
    def __getitem__(self, idx):
        tile, t  = self.samples[idx]
        basepath = os.path.join(self.root_dir, tile)

        static_arr = xr.open_dataset(os.path.join(basepath, "static.nc"))[self.static_vars].to_array().values

        dyn_arr = xr.open_dataset(os.path.join(basepath, "dynamic.nc")).isel(time=t).to_array().values

        target_arr = xr.open_dataset(os.path.join(basepath, "target.nc")).isel(time=t).to_array().values

        static_tensor = self.norm_static(torch.as_tensor(static_arr, dtype=torch.float32))

        dyn_vector = self.norm_dyn(
                            torch.as_tensor(dyn_arr[:,0,0], dtype=torch.float32)  # Instead of 2D, just vector; fill it out with unet class
                        )

        target_tensor = torch.as_tensor(target_arr, dtype=torch.float32)

        return static_tensor, dyn_vector, target_tensor
