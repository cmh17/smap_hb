#!/usr/bin/env python3
# plot_unet_time_series.py ---------------------------------------------------
"""
Rough diagnostic plots for UNet predictions vs. targets
-------------------------------------------------------

Creates three PNG files in the same directory as --out‑prefix:

  * <prefix>_spatial_mean.png
  * <prefix>_rmse.png
  * <prefix>_step_scatter.png

Example usage:
python Carrie/cypress_creek/scripts/plot_unet_time_series.py \
       --workspace /work/nv25/Carrie/cypress_creek \
       --pred      /work/nv25/Carrie/cypress_creek/models/model_outputs/unet8_std_predictions_20250505_173239.npy \
       --tgt       /work/nv25/Carrie/cypress_creek/models/model_outputs/unet8_std_targets_20250505_173239.npy \
       --out-prefix /work/nv25/Carrie/cypress_creek/figures/unet8_2017
"""
import argparse, pathlib, os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def read_times(workspace, dyn_nc_rel, n_steps):
    """
    Returns 1D array with datetime64 validation timestamps
    (2017, stride 10) since that's how I set up the train/val split.
    """
    ds   = xr.open_dataset(pathlib.Path(workspace) / dyn_nc_rel)
    allt = ds.time.values
    mask = (allt.astype("datetime64[Y]").astype(int) + 1970) == 2017
    return allt[mask][::10][:n_steps]

def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--workspace", required=True)
    argp.add_argument("--pred",      required=True, help="*.npy predictions")
    argp.add_argument("--tgt",       required=True, help="*.npy targets")
    argp.add_argument("--out-prefix",required=True, help="prefix for *.png files")
    argp.add_argument("--dynamic-nc",
                      default="data/combined_output/tiles2/tile_0_0/dynamic.nc")
    argp.add_argument("--steps", type=int, default=23,
                      help="validation steps per tile (default: 23)")
    argp.add_argument("--tiles", type=int, default=100,
                      help="how many 360×360 tiles (default: 100)")
    args = argp.parse_args()

    # Load
    pred = np.load(args.pred)   # shape: (N, H, W)
    tgt  = np.load(args.tgt)

    # Get shape
    N, H, W = pred.shape
    
    # Check directory tiles count
    import glob, os
    tiles_list = glob.glob(os.path.join(args.workspace,
                                        "data/combined_output/tiles2", "tile_*"))
    TILES = len(tiles_list)
    STEPS = N // TILES
    if TILES * STEPS != N:
        raise ValueError(f"Cannot factor {N} samples into tiles×steps (got {TILES}×{STEPS})")

    # Reshape into (tiles, steps, H, W)
    pred = pred.reshape(TILES, STEPS, H, W)
    tgt  = tgt .reshape(TILES, STEPS, H, W)

    # Make dir if needd
    from pathlib import Path
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Per‐step spatial mean
    pred_step_mean = pred.mean(axis=(0,2,3))
    tgt_step_mean  = tgt .mean(axis=(0,2,3))

    # Per‐step RMSE
    step_rmse = np.sqrt(((pred - tgt)**2).mean(axis=(0,2,3)))

    # timestamps
    times = read_times(args.workspace, args.dynamic_nc, args.steps)
    # times must also have length == STEPS:
    times = read_times(args.workspace, args.dynamic_nc, STEPS)

    # Spatial mean fig
    plt.figure(figsize=(7,4))
    plt.plot(times, tgt_step_mean,  label="observed", marker='o')
    plt.plot(times, pred_step_mean, label="predicted", marker='o')
    plt.xlabel("Date"); plt.ylabel("Spatial mean")
    plt.title("UNet | spatial mean per 360×360 tile (validation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_spatial_mean.png", dpi=300)
    plt.close()

    # RMSE vs time fig
    plt.figure(figsize=(7,4))
    plt.plot(times, step_rmse, marker='o', color="crimson")
    plt.xlabel("Date"); plt.ylabel("RMSE")
    plt.title("UNet | per‑step RMSE (spatial)")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_rmse.png", dpi=300)
    plt.close()

    # Scatter of step means fig... probably a better vis than this
    plt.figure(figsize=(4,4))
    plt.scatter(tgt_step_mean, pred_step_mean, c='steelblue')
    _lims = [min(tgt_step_mean.min(), pred_step_mean.min()),
             max(tgt_step_mean.max(), pred_step_mean.max())]
    plt.plot(_lims, _lims, '--k', linewidth=1)
    plt.xlabel("Observed mean"); plt.ylabel("Predicted mean")
    plt.title("UNet step‑wise mean (scatter)")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_step_scatter.png", dpi=300)
    plt.close()

    print("Figures saved with prefix ", args.out_prefix)

if __name__ == "__main__":
    main()
