import os
import sys
sys.path.append(os.path.abspath('model_scripts'))
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt

from data_loader_multi_timestep import CombinedDataset
from unet import UNetFusion

import xarray as xr
from datetime import datetime
import numpy as np

def make_figures(predictions_npy_path, 
                 targets_npy_path, 
                 predictions_fig_out, 
                 times,
                 time_index,
                 train_time_inds,
                 test_time_inds,
                 targets_fig_out=None,
                 mean_fig_out=None):
    
    unet7_predictions = np.load(predictions_npy_path)
    targets = np.load(targets_npy_path)
    
    # Split into tiles with all time steps
    unet7_prediction_scene = np.empty([len(test_time_inds), 3600, 3600])
    target_scene = np.empty([len(test_time_inds), 3600, 3600])
    
    # Recreate whole scenes from 100 tiles with 23 time steps
    for i in range(2300):
    #     print(i,(i+23))
        tile_num = i // 23
        tile_x = tile_num % 10
        tile_y = tile_num // 10
        time_index = i % 23
    #     print(time_index, tile_x, tile_y)
        unet7_prediction_scene[time_index, tile_x*360:(tile_x+1)*360, tile_y*360:(tile_y+1)*360] = unet7_predictions[i]
        target_scene[time_index, tile_x*360:(tile_x+1)*360, tile_y*360:(tile_y+1)*360] = targets[i]
    print("Combined tiles.")
    
    vmin, vmax = 0, target_scene.max()
    ticks = np.linspace(vmin, vmax, 6) 
    
    plt.figure()
    im = plt.imshow(
            unet7_prediction_scene[4],
            vmin=vmin, vmax=vmax, origin='lower')
    cb = plt.colorbar(im, ticks=ticks)
    cb.ax.tick_params(labelsize=10)
    plt.title(f"UNet prediction {times[4]}")
    plt.savefig(predictions_fig_out, dpi=300)
    plt.close()

    if targets_fig_out is not None:
        plt.figure()
        im = plt.imshow(
                target_scene[4],
                vmin=vmin, vmax=vmax, origin='lower')
        cb = plt.colorbar(im, ticks=ticks)
        cb.ax.tick_params(labelsize=10)
        plt.title(f"Target {times[4]}")
        plt.savefig(targets_fig_out, dpi=300)
        plt.close()

    if mean_fig_out is not None:
        target_scene_mean = np.mean(target_scene, axis=0)
        plt.figure()
        im = plt.imshow(
                target_scene_mean,
                vmin=vmin, vmax=vmax, origin='lower')
        cb = plt.colorbar(im, ticks=ticks)
        cb.ax.tick_params(labelsize=10)
        plt.title(f"Target Scene Mean")
        plt.savefig(mean_fig_out, dpi=300)
        plt.close()
        
    print("Saved figure(s).")
    
def main():
    # Come back and make this take the prediction and target paths as inputs
    workspace = os.path.dirname(os.getcwd())

    ds = xr.open_dataset(os.path.join(workspace, "data", "combined_output", "tiles2", "tile_0_0", "dynamic.nc"))
    
    times = ds.time.values

    time_inds = list(range(0,(len(ds.time.values))))

    years = ds.time.values.astype('datetime64[Y]').astype(int) + 1970

    train_time_inds = list(range(list(years).index(2016),list(years).index(2017), 10))

    test_time_inds = list(range(list(years).index(2017),list(years).index(2018), 10))

    predictions_npy_path = os.path.join(workspace, "models", "model_outputs", "unet8_std_predictions_20250505_173239.npy")
    targets_npy_path = os.path.join(workspace, "models", "model_outputs", "unet8_std_targets_20250505_173239.npy")
    
    predictions_fig_out = os.path.join(workspace, "models", "model_outputs", "unet8_std_predictions_20250505_173239.png")
    targets_fig_out = os.path.join(workspace, "models", "model_outputs", "unet8_std_targets_20250505_173239.png")
    mean_fig_out = os.path.join(workspace, "models", "model_outputs", "unet8_std_mean_20250505_173239.png")
    
#     print(predictions_fig_out)
    make_figures(predictions_npy_path, 
                 targets_npy_path, 
                 predictions_fig_out,
                 times,
                 10,
                 train_time_inds,
                 test_time_inds,
                 targets_fig_out,
                 mean_fig_out)


if __name__ == "__main__":
    main()


    

