#!/usr/bin/env python3
# make_static_cache.py -------------------------------------------------
"""
Convert each tile’s static.nc to static_<dtype>.npy  (C × H × W).
Creates / updates static_manifest.json mapping tile → relative npy path.

If a legacy file static_f16.npy is found and --dtype float16 is used,
it is renamed to the new pattern.  If --dtype is not float16, the old
file is left untouched unless you pass --clean-legacy.
"""
import os, xarray as xr, numpy as np, json, argparse, tqdm, shutil

def list_tiles(root):
    return sorted(
        d for d in os.listdir(root)
        if d.startswith("tile_") and os.path.isdir(os.path.join(root, d))
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",   required=True, help="tiles2 directory")
    p.add_argument("--dtype",  default="float16",
                   choices=["float16", "float32", "float64"],
                   help="NumPy dtype for saved array")
    p.add_argument("--vars",   nargs="+", default=None,
                   help="Subset of static variables to keep")
    p.add_argument("--clean-legacy", action="store_true",
                   help="Delete old static_f16.npy if dtype != float16")
    args = p.parse_args()

    root = args.root
    dtype = np.dtype(args.dtype)  # creates np.float16, etc.
    suffix = f"static_{args.dtype}.npy"
    manifest = {}
    
    tiles = list_tiles(root)
    order_path = os.path.join(root, "static_var_order.json")
    
    # Check first tile to get order of variables and save a json of the info
    if not os.path.exists(order_path):
        sample_tile = tiles[0]
        s_nc = os.path.join(root, sample_tile, "static.nc")
        with xr.open_dataset(s_nc) as ds:
            var_order = args.vars or list(ds.data_vars)
        with open(order_path, "w") as f:
            json.dump(var_order, f)
        print(f"Saved variable order for {len(var_order)} channels to {order_path}")
    

    for tile in tqdm.tqdm(list_tiles(root), desc="static to npy"):
        tile_dir   = os.path.join(root, tile)
        out_path   = os.path.join(tile_dir, suffix)
        legacy_f16 = os.path.join(tile_dir, "static_f16.npy")

        # Rename for consistency _-_--__
        if args.dtype == "float16" and os.path.exists(legacy_f16):
            if not os.path.exists(out_path):
                print(f"Renaming {legacy_f16} → {out_path}")
                shutil.move(legacy_f16, out_path)
            else:
                print(f"Found both {legacy_f16} and new file; keeping new.")
            manifest[tile] = os.path.relpath(out_path, root)
            continue
        elif args.clean_legacy and os.path.exists(legacy_f16):
            print(f"Deleting legacy file {legacy_f16}")
            os.remove(legacy_f16)

        # Create new if needed
        if not os.path.exists(out_path):
            s_nc = os.path.join(tile_dir, "static.nc")
            with xr.open_dataset(s_nc) as ds:
                arr = (
                    ds[args.vars or list(ds.data_vars)]
                    .to_array()
                    .astype(dtype, copy=False)
                    .valueas  # (C, H, W)
                )
            np.save(out_path, arr, allow_pickle=False)

        manifest[tile] = os.path.relpath(out_path, root)

    man_path = os.path.join(root, "static_manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest with {len(manifest)} tiles to {man_path}")

if __name__ == "__main__":
    main()
