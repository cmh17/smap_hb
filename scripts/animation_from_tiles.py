#!/usr/bin/env python3
# make_unet_gif.py -----------------------------------------------------------
"""
Create an animated GIF that shows UNet predictions vs. targets for every
validation time‑step.

Example usage:
python make_unet_gif.py \
  --workspace /work/nv25/Carrie/cypress_creek \
  --pred npy/unet7_std_predictions_20250505_153716.npy \
  --tgt  npy/unet7_std_targets_20250505_153716.npy \
  --out  gif/unet7_2017_validation.gif
"""

import argparse, pathlib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable   # color bar helper

def make_scene_cube(pred_np: np.ndarray,
                    tgt_np : np.ndarray,
                    n_tiles : int = 100,
                    steps   : int = 23,
                    tile_sz : int = 360):
    """
    Re‑assemble (n_tiles*steps, H_tile, W_tile) to (steps, H, W).
    """
    grid  = int(np.sqrt(n_tiles))
    H_big = grid * tile_sz
    out_p = np.empty((steps, H_big, H_big), dtype=pred_np.dtype)
    out_t = np.empty_like(out_p)

    for flat in range(n_tiles * steps):
        tile_i  = flat // steps              # 0 to 99
        t_i     = flat  % steps              # 0 to 22
        gx, gy  = divmod(tile_i, grid)       # (row, col)

        y0, y1 = gy * tile_sz, (gy + 1) * tile_sz
        x0, x1 = gx * tile_sz, (gx + 1) * tile_sz

        out_p[t_i, y0:y1, x0:x1] = pred_np[flat]
        out_t[t_i, y0:y1, x0:x1] = tgt_np [flat]

    return out_p, out_t

def main(argv=None):
    p = argparse.ArgumentParser(description="Animate UNet predictions vs targets")
    p.add_argument("--workspace", required=True,
                   help="Path to project root (.../cypress_creek)")
    p.add_argument("--pred", required=True, help="*.npy file with model predictions")
    p.add_argument("--tgt",  required=True, help="*.npy file with target data")
    p.add_argument("--out",  required=True, help="Output *.gif filename")
    p.add_argument("--dynamic-nc",
                   default="data/combined_output/tiles2/tile_0_0/dynamic.nc",
                   help="Path within workspace to a dynamic.nc (for timestamps)")
    p.add_argument("--fps", type=int, default=3, help="Frames per second")
    p.add_argument("--tick-count", type=int, default=6, help="# colour‑bar ticks")
    args = p.parse_args(argv)

    ws_path  = pathlib.Path(args.workspace)
    pred_np  = np.load(args.pred)
    tgt_np   = np.load(args.tgt)

    # Get geometry
    n_samples, H, _ = pred_np.shape
    steps = 23 # Change this to not be hardcoded later
    n_tiles = n_samples // steps
    tile_size = H  # 360

    if n_tiles * steps != n_samples:
        raise ValueError("pred array does not factor into tiles × steps")

    # Timestamps
    dyn_path  = ws_path / args.dynamic_nc
    times_full = xr.open_dataset(dyn_path).time.values
    # Change this to be an input later; maybe include times in model output
    mask_2017  = (times_full.astype("datetime64[Y]").astype(int) + 1970) == 2017
    time_inds  = np.arange(len(times_full))[mask_2017][::10]
    time_lbls  = times_full[time_inds]

    # Recreate scenes from tiles
    scene_pred, scene_tgt = make_scene_cube(pred_np, tgt_np,
                                            n_tiles=n_tiles,
                                            steps=steps,
                                            tile_sz=tile_size)

    vmin, vmax = 0, scene_tgt.max()
    ticks      = np.linspace(vmin, vmax, args.tick_count)

    # Draw each frame
    frames = []
    for ti, tlabel in enumerate(time_lbls):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        im_pred = axes[0].imshow(scene_pred[ti], vmin=vmin, vmax=vmax,
                                 origin='lower')
        axes[0].set_title(f"Prediction\n{np.datetime_as_string(tlabel, 'D')}")

        im_tgt  = axes[1].imshow(scene_tgt[ti],  vmin=vmin, vmax=vmax,
                                 origin='lower')
        axes[1].set_title("Target")

        for a in axes:
            a.set_xticks([]); a.set_yticks([])

        # add color bar -- trying to give space but doesn't work very well yet
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("bottom", size="5%", pad=0.6)  # inches
        cb = fig.colorbar(im_tgt, cax=cax, orientation='horizontal',
                               ticks=ticks)
        cb.ax.tick_params(labelsize=8)

        fig.tight_layout()

        # Turn figure to rgb array
        fig.canvas.draw_idle()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    # Save as gif
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path,
                frames,
                fps=args.fps,
                loop=0) 
    print("Saved GIF to", out_path)

if __name__ == "__main__":
    main()
