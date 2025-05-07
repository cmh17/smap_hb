import os, json, torch, xarray as xr, numpy as np
from torch.utils.data import Dataset
from make_static_cache import list_tiles

# Same idea as previous data loaders,
# but this time use static npy files instead of nc
# and load them all together instead of loading again for each piece of training data


class ZScore:
    def __init__(self, mean, std, eps=1e-7):
        self.mean = torch.as_tensor(mean)
        self.std  = torch.as_tensor(std) + eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        view = (-1,) + (1,)*(x.ndim-1)
        mean = self.mean.to(x.device)
        std  = self.std.to(x.device)
        return (x - mean.view(*view)) / std.view(*view)

class CombinedDataset(Dataset):
    def __init__(self, root_dir, static_stats_file, dynamic_stats_file,
                 static_vars=None, dynamic_vars=None,
                 time_inds=None, listoftiles=None,
                 mmap_static=True, dtype=np.float16,
                 cache_static=True, cache_device="cpu",
                 **kw):
        
        self.root_dir = root_dir
        self.dtype = dtype
        self.mmap_static = mmap_static

        self.cache_static = cache_static            # turn on/off
        self.cache_device = torch.device(cache_device)
        self._static_cache = {} 
        self._dyn_cache = {}
        self._tgt_cache = {}
        
        # access the manifest to get npy locations
        if mmap_static:
            with open(os.path.join(root_dir, "static_manifest.json")) as f:
                self.static_manifest = json.load(f)
                
        with open(os.path.join(root_dir, "target_manifest.json")) as f:
                self.target_manifest = json.load(f)

        # get var names if a subset wasn't specified
        if static_vars is None:
            sample_tile = listoftiles[0] if listoftiles else list_tiles(root_dir)[0]
            with xr.open_dataset(os.path.join(root_dir, sample_tile, "static.nc")) as ds:
                static_vars = list(ds.data_vars)

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
        idx_map = [file_order.index(v) for v in self.static_vars]  # get index of each of the static vars in the json
        self._static_idx = np.array(idx_map, dtype=np.int64)

        # Tile timestamp list -- retrieve from one of the tiles if not given
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
        tens = torch.from_numpy(arr)
        tens = self.norm_static(tens)

        # Push to GPU & keep there
        if self.cache_device.type == "cuda":
            tens = tens.to(self.cache_device, dtype=torch.float16,
                           non_blocking=True)

        if self.cache_static:
            self._static_cache[tile] = tens

        return tens
    
    def __getitem__(self, idx):
        tile, t = self.samples[idx]

        static_tensor = self._load_static(tile)
        
        if tile not in self._dyn_cache:
            dpath = os.path.join(self.root_dir, tile, "dynamic_T2_float32.npy")
            cpu_arr  = np.load(dpath, mmap_mode="r").copy()          # (T,2)
            gpu_tens = torch.from_numpy(cpu_arr).float().to("cuda", non_blocking=True)  # (T,2)
            self._dyn_cache[tile] = gpu_tens

        dyn_vector = self.norm_dyn(self._dyn_cache[tile][t])         # (2,) on cuda

        if tile not in self._tgt_cache:
            tpath = os.path.join(self.root_dir, tile, "target_THW_float16.npy")
            npmap = np.load(tpath, mmap_mode="r")         # read‑only
            self._tgt_cache[tile] = torch.from_numpy(npmap).pin_memory()  # CPU pinned

        target_tensor = (
            self._tgt_cache[tile][t]      # (H,W)
            .unsqueeze(0)                 # (1,H,W) to match prediction
            .to("cuda", non_blocking=True)
        )

        return static_tensor, dyn_vector, target_tensor
    