# make_dynamic_cache.py --------------------------------------------
import os, xarray as xr, numpy as np, argparse, tqdm
from make_target_cache import list_tiles      # reuse the helper

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--dtype", default="float32")
args = p.parse_args()

for tile in tqdm.tqdm(list_tiles(args.root)):
    dyn_nc = os.path.join(args.root, tile, "dynamic.nc")
    out = os.path.join(args.root, tile, f"dynamic_{args.dtype}.npy")
    if os.path.exists(out):
        continue
    with xr.open_dataset(dyn_nc) as ds:
        arr = ds.to_array().astype(args.dtype, copy=False).values  # (time, 2)
    np.save(out, arr, allow_pickle=False)
