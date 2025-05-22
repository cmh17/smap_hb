#!/usr/bin/env python3
"""
Filename: data_loader_uet10.py
Author: Caroline Hashimoto
Date: 2025-05-22
Description: This data loader takes in a directory of directories 
for each tile in the study area. Each tile directory has dynamic, 
static, and target numpy files. This loader differs from previous
because it transforms the target to be the anomaly compared to the
mean from the dynamic, coarse-resolution soil moisture.
"""

import os, json, torch, xarray as xr, numpy as np
from torch.utils.data import Dataset
from make_static_cache import list_tiles

class ZScore:
    def __init__(self, mean, std, eps=1e-7):
        self.mean = torch.as_tensor(mean)
        self.std  = torch.as_tensor(std) + eps

    def __call__(self, x: torch.Tensor):
        view = (-1,) + (1,)*(x.ndim-1)
        mean = self.mean.to(x.device)
        std  = self.std.to(x.device)
        return (x - mean.view(*view)) / std.view(*view) # asterisk unpacks list

class CombinedDataset(Dataset):
    def __init__(self, root_dir, static_stats_file, dynamic_stats_file,
                 static_vars=None, dynamic_vars=None,
                 time_inds=None, listoftiles=None, dtype=np.float16,
                 cache_static=True, cache_device="cpu", include_twi=True,
                 **kw):
        
        self.root_dir = root_dir
        self.dtype = dtype
        self.include_twi = include_twi

        self.cache_static = cache_static  # Boolean
        self.cache_device = torch.device(cache_device)
        self._static_cache = {} 
        self._dyn_cache = {}
        self._dyn_vec_cache = {}
        self._tgt_cache = {}
        self._anom_cache = {} # Cache the anomalies so you don't have to recompute every time
        
        # Access the manifest to get npy locations
        with open(os.path.join(root_dir, "static_manifest.json")) as f:
                self.static_manifest = json.load(f)
                
#         with open(os.path.join(root_dir, "target_manifest.json")) as f: # Target manifest is out of date rn, so ignore
#                 self.target_manifest = json.load(f)

        # get var names if a subset wasn't specified
        if static_vars is None:
            sample_tile = listoftiles[0] if listoftiles else list_tiles(root_dir)[0]
            with xr.open_dataset(os.path.join(root_dir, sample_tile, "static.nc")) as ds:
                static_vars = list(ds.data_vars)
                
                # TWI isn't in the static file, so if it's included, add it here too
                if self.include_twi:
                    static_vars.append("twi")

        if dynamic_vars is None:
            with xr.open_dataset(os.path.join(root_dir,
                                              (listoftiles or list_tiles(root_dir))[0],
                                              "dynamic.nc")) as ds:
                dynamic_vars = list(ds.data_vars)

        # Add them as attributs of the class
        self.static_vars, self.dynamic_vars = static_vars, dynamic_vars
        
        # Get indices of static data for chosen static vars
        with open(os.path.join(root_dir, "static_var_order.json")) as f:
            file_order = json.load(f)
#         print("Debugging, file order:\n", file_order, "\nstatic vars:", self.static_vars)
        idx_map = [file_order.index(v) for v in self.static_vars] # get index of each of the static vars in the json
        self._static_idx = np.array(idx_map, dtype=np.int64)

        # Tile timestamp list -- retrieve from tiles directory if not given
        self.tiles = listoftiles or list_tiles(root_dir)
        if time_inds is None:
            with xr.open_dataset(os.path.join(root_dir, self.tiles[0],
                                              "dynamic.nc")) as ds:
                time_inds = range(len(ds.time))
        self.samples = [(tile, t) for tile in self.tiles for t in time_inds]

        # Get norm stats for static and dynamic, saved earlier in json
        with open(static_stats_file)  as f: s_stats = json.load(f)
        with open(dynamic_stats_file) as f: d_stats = json.load(f)

        # Use helper function to normalize with dataset-specific stats for training data
        self.norm_static = ZScore([s_stats[v]['mean'] for v in self.static_vars],
                                  [s_stats[v]['std']  for v in self.static_vars])

        self.norm_dyn    = ZScore([d_stats[v]['mean'] for v in self.dynamic_vars],
                                  [d_stats[v]['std']  for v in self.dynamic_vars])
        
    def __len__(self): return len(self.samples)

    def _load_static(self, tile):
        if self.cache_static and tile in self._static_cache:
            return self._static_cache[tile] # reuse if it's already there

        # Otherwise load it from the path specified in the manifest
        path = os.path.join(self.root_dir, self.static_manifest[tile])
        arr = np.load(path, mmap_mode="r")[self._static_idx]
        
        # No longer needed
#         # Band-aid for adding TWI -- consider finding another way to do this
#         if self.include_twi:
#             twi = np.load(os.path.join(self.root_dir, tile, "twi_float16.npy"))
#             arr = np.concatenate([arr, twi[None, ...]], axis=0)
            
        tens = torch.from_numpy(arr)
        tens = self.norm_static(tens)

        # Push to GPU & keep there
        if self.cache_device.type == "cuda":
            tens = tens.to(self.cache_device, dtype=torch.float16,
                           non_blocking=True)

        self._static_cache[tile] = tens
        return tens
    
    def _prepare_anomaly_cube(self, tile):
        """
        Compute the anomaly cube (T,H,W) once for the given tile and keep it
        in RAM as float16. This is where the subtraction happens, so
        __getitem__ only has to index.
        """
        # dynamic (T,2); currently in RAM, not GPU yet
        dyn_np = self._dyn_cache[tile][:, 1].cpu().numpy()
        # target (T,H,W), float16
        
        # New cache for normalized dyn vec
        if tile not in self._dyn_vec_cache:
            self._dyn_vec_cache[tile] = self.norm_dyn(self._dyn_cache[tile].T).T.to(self.cache_device, dtype=torch.float16, non_blocking=True)

        tgt = np.load(os.path.join(self.root_dir, tile, "target_THW_float16.npy"), mmap_mode="r")
        
        anom = (tgt.astype(np.float32) -
                dyn_np[:, None, None].astype(np.float32)).astype(np.float16)
        
        # On CPU with memory pinned to make it fast
        self._anom_cache[tile] = torch.from_numpy(anom).pin_memory()
  
    def __getitem__(self, idx):
        
        tile, t = self.samples[idx]
        
        if tile not in self._static_cache:
            self._load_static(tile)
            
        if tile not in self._dyn_cache:
            dpath = os.path.join(self.root_dir, tile, "dynamic_T2_float32.npy")
            cpu_arr  = np.load(dpath, mmap_mode="r").copy()
            gpu_tens = torch.from_numpy(cpu_arr).float().to("cuda", non_blocking=True)
            self._dyn_cache[tile] = gpu_tens
            
        if tile not in self._dyn_vec_cache:
            self._dyn_vec_cache[tile] = self.norm_dyn(self._dyn_cache[tile].T).T.to(self.cache_device, dtype=torch.float16, non_blocking=True) # On GPU
            
        if tile not in self._anom_cache:
            self._prepare_anomaly_cube(tile) # On CPU
        
        static_tensor = self._static_cache[tile]
        dyn_vector = self._dyn_vec_cache[tile][t]
        anom_map = self._anom_cache[tile][t].unsqueeze(0).to(self.cache_device, dtype=torch.float16, non_blocking=True) # Move to GPU
        
        return static_tensor, dyn_vector, anom_map

