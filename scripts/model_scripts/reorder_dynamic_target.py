# reorder_dynamic_target.py
import numpy as np, os, tqdm, argparse

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
args = p.parse_args()

def list_tiles(root):
    return sorted(
        d for d in os.listdir(root)
        if d.startswith("tile_") and os.path.isdir(os.path.join(root, d))
    )

for tile in tqdm.tqdm(list_tiles(args.root)):
    tdir = os.path.join(args.root, tile)
    # dynamic  (2,T,1,1) to (T,2)
    d_old = os.path.join(tdir, "dynamic_float32.npy")
    d_new = os.path.join(tdir, "dynamic_T2_float32.npy")
    if not os.path.exists(d_new):
        dyn = np.load(d_old, mmap_mode="r")
        np.save(d_new, dyn.transpose(1,0,2,3).reshape(dyn.shape[1], 2), allow_pickle=False)
    # target   (1,T,H,W) to (T,H,W)
    t_old = os.path.join(tdir, "target_float16.npy")
    t_new = os.path.join(tdir, "target_THW_float16.npy")
    if not os.path.exists(t_new):
        tgt = np.load(t_old, mmap_mode="r")
        np.save(t_new, tgt[0], allow_pickle=False)          # drop channel dim
