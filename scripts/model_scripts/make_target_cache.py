# make_target_cache.py
import os, xarray as xr, numpy as np, tqdm, argparse, json

def list_tiles(root):
    """
    Make sure to just get the tile folders and not the jsons or any hidden folders.
    """
    return sorted(
        d for d in os.listdir(root)
        if d.startswith("tile_")                # ← optional but convenient
        and os.path.isdir(os.path.join(root, d))
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()

    manifest = {}
    dtype = np.dtype(args.dtype)

    for tile in tqdm.tqdm(list_tiles(args.root)):
        tgt_nc = os.path.join(args.root, tile, "target.nc")
        npy    = os.path.join(args.root, tile, f"target_{args.dtype}.npy")
        if not os.path.exists(npy):
            with xr.open_dataset(tgt_nc) as ds:
                arr = ds.to_array().astype(dtype, copy=False).values
            np.save(npy, arr, allow_pickle=False)
        manifest[tile] = os.path.relpath(npy,args.root)

    with open(os.path.join(args.root, "target_manifest.json"), "w") as f:
        json.dump(manifest, f)
        
if __name__ == "__main__":
    main()
