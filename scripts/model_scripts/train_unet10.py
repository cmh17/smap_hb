import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import xarray as xr
import time

from data_loader_unet10 import CombinedDataset
from unet10 import UNetFusionExplicit

from torch.cuda.amp import autocast # Try using half precision when safe
from torch.cuda.amp import GradScaler
scaler = GradScaler() 

import random, numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.use_deterministic_algorithms(True) # Remove for speed
torch.backends.cudnn.benchmark = True # "cudnn will look for the optimal set of algorithms for that particular configuration" -- can make it faster
# Unless it changes configuration every time, in which case it gets worse

workspace = os.path.dirname(os.path.dirname(os.getcwd()))
root_dir = f"{workspace}/data/combined_output/tiles2"
os.makedirs(os.path.join(workspace, "models", "model_outputs"), exist_ok=True)

loading_start_time = time.time()

ds = xr.open_dataset(os.path.join(workspace, "data", "combined_output", "tiles2", "tile_0_0", "dynamic.nc"))
time_inds = list(range(0, (len(ds.time.values))))

years = ds.time.values.astype("datetime64[Y]").astype(int) + 1970

# Take time slices from 2016-2018, stepping by 10 (by days, but don't have all days)
# time_inds = list(range(list(years).index(2016), list(years).index(2018), 10))

train_time_inds = list(range(list(years).index(2015), list(years).index(2019), 10)) # Starts in April, but that's fine
test_time_inds = list(range(list(years).index(2019), len(years), 10))

# Including topographic wetness index since it's expected to be a good predictor
include_twi = True

# train_tiles = []
# test_tiles = []

# # If I was doing the checkerboard thing
# for i in range(10):
#     for j in range(10):
#         new_tile = f"tile_{i}_{j}"
#         if (i + j) % 2 == 0:
#             train_tiles.append(new_tile)
#         else:
#             test_tiles.append(new_tile)

static_vars = ['bd_0_5',
               'clay_0_5',
               'hb_0_5',
               'ksat_0_5',
               'n_0_5',
               'om_0_5',
               'ph_0_5',
               'sand_0_5',
               'silt_0_5',
               'theta_r_0_5',
               'theta_s_0_5',
               'commercial',
               'cropland',
               'elevation',
               'exurban_high',
               'exurban_low',
               'grazing',
               'industrial',
               'institutional',
               'mining_barren_land',
               'natural_water',
               'parks_golf_courses',
               'pasture',
               'recreation_conservation',
               'reservoirs_canals',
               'suburban',
               'timber',
               'transportation',
               'urban_high',
               'urban_low',
               'wetlands',
               'twi']

training_set = CombinedDataset(root_dir, time_inds=train_time_inds, static_vars=static_vars,
                              static_stats_file=os.path.join(workspace, "data", "combined_output", "static_norm_stats.json"),
                              dynamic_stats_file=os.path.join(workspace, "data", "combined_output", "dynamic_norm_stats.json"),
                              cache_static=True,cache_device="cuda", include_twi=include_twi)
validation_set = CombinedDataset(root_dir, time_inds=test_time_inds, static_vars=static_vars,
                                 static_stats_file=os.path.join(workspace, "data", "combined_output", "static_norm_stats.json"),
                                 dynamic_stats_file=os.path.join(workspace, "data", "combined_output", "dynamic_norm_stats.json"),
                                 cache_static=True,cache_device="cuda", include_twi=include_twi)

batch_size  = 8 # 2300 tiles
num_workers = 0 # So far no luck resolving memory error
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

# # Check some data
# for static, dyn, target in training_loader:
#     print("static shape: ", static.shape) 
#     print("dynamic shape:", dyn.shape)
#     print("target shape: ", target.shape)

#     print("static: ", static) 
#     print("dynamic:", dyn)
#     print("target: ", target) 
#     break

loading_end_time = time.time()

loading_time = loading_end_time - loading_start_time
print(f"Loading data took {loading_time:.2f} seconds")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_static = len(static_vars) #+ (1 if include_twi else 0) # no longer needed since added twi to rest of static npy (not nc though)
model = UNetFusionExplicit(in_channels_static=n_static, out_channels=1).to(device) # 32 in-channels
if torch.cuda.device_count() > 1: # Try to use multiple GPUs
     model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Try hybrid loss function
mae_fn = torch.nn.L1Loss()
mse_fn = torch.nn.MSELoss()
alpha  = 0.6 # Tune this

def combined_loss(pred, target):
    return alpha * mse_fn(pred, target) + (1 - alpha) * mae_fn(pred, target)

# Just using MSE this time
loss_fn = mse_fn

def train_one_epoch(epoch, writer=None):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    start_epoch_time = time.time()
    
    start = time.perf_counter()
    
    data_iter = iter(training_loader)
    for step in range(len(training_loader)):     
        start = time.perf_counter()
        static_maps, dyn_vec, target = next(data_iter)
        # Move to CUDA, or whatever device is being used
        static_maps = static_maps.to(device, non_blocking=True)
        dyn_vec = dyn_vec.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        loader_ms = (time.perf_counter() - start) * 1e3  
        
        # For timing
        torch.cuda.synchronize()
        start_gpu = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            pred = model(static_maps, dyn_vec)                 
            loss = loss_fn(pred, target) 
            mse_loss = mse_fn(pred, target)
            mae_loss = mae_fn(pred, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 
        
        torch.cuda.synchronize()
        gpu_t = time.perf_counter() - start_gpu
        
        if step < 20: # Print out for first 20 batches
            print(f"step {step:<3} loader {loader_ms:6.1f} ms  gpu {gpu_t*1e3:6.1f} ms")
        start = time.perf_counter()

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_mae += mae_loss.item()

    epoch_time = time.time() - start_epoch_time
    print(f"Epoch {epoch} finished in {epoch_time:.2f} seconds")

    # Average loss
    avg_loss = total_loss / len(training_loader)
    print(f"Epoch {epoch}, average train loss={avg_loss:.4f}")
    
    avg_mae = total_mae / len(training_loader)
    avg_mse = total_mse / len(training_loader)

    if writer is not None:
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Timing/train_epoch_seconds", epoch_time, epoch)
        writer.add_scalar("Metric/train_MAE", avg_mae, epoch)
        writer.add_scalar("Metric/train_MSE", avg_mse, epoch)

    return avg_loss, avg_mse, avg_mae

def validate_one_epoch(epoch, writer=None):
    model.eval()
    total_vloss = 0.0
    val_mse = 0.0
    val_mae = 0.0
    start_val_time = time.time()

    with torch.no_grad():
        data_iter = iter(validation_loader)
        for step in range(len(validation_loader)):
            s, d, t = next(data_iter)
            with autocast():
                pred  = model(s, d)
                vloss = loss_fn(pred, t)
                mse = mse_fn(pred,t)
                mae = mae_fn(pred,t)
            total_vloss += vloss.item()
            val_mse += mse.item()
            val_mae += mae.item()

    val_time = time.time() - start_val_time
    print(f"Validation for epoch {epoch} took {val_time:.2f} seconds")

    avg_vloss = total_vloss / len(validation_loader)
    print(f"Validation Mixed Loss: {avg_vloss:.4f}")
    
    avg_mse = val_mse / len(validation_loader)
    avg_mae = val_mae / len(validation_loader)

    if writer is not None:
        writer.add_scalar("Loss/val", avg_vloss, epoch)
        writer.add_scalar("Timing/val_epoch_seconds", val_time, epoch)
        writer.add_scalar("Metric/val_MAE", avg_mae, epoch)
        writer.add_scalar("Metric/val_MSE", avg_mse, epoch)
        
    return avg_vloss, avg_mse, avg_mae

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=f"{workspace}/models/runs/unet10_trainer_{timestamp}")

num_epochs = 100
best_vloss = float("inf") # set to max to start

train_losses = []
train_mse_list = []
train_mae_list = []
val_losses = []
val_mse_list = []
val_mae_list = []

for epoch in range(num_epochs):
    print(f"--- EPOCH {epoch+1}/{num_epochs} ---")

    # Train
    avg_loss, avg_mse, avg_mae = train_one_epoch(epoch, writer=writer)
    train_losses.append(avg_loss)
    train_mse_list.append(avg_mse)
    train_mae_list.append(avg_mae)

    # Validate
    avg_vloss, avg_val_mse, avg_val_mae = validate_one_epoch(epoch, writer=writer)
    val_losses.append(avg_vloss)
    val_mse_list.append(avg_val_mse)
    val_mae_list.append(avg_val_mae)

    print(f"Train Loss: {avg_loss:.6f}, Val Loss: {avg_vloss:.6f}")

    # Save best model, keep track of best validation loss to determine if this epoch does better than prev best
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"{workspace}/models/model_outputs/unet10_std_model_{timestamp}_epoch{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved new best model to {model_path}")

end_time = time.time()
print("Time to load, train, and validate: ", end_time - loading_start_time)

plt.figure()
plt.plot(train_losses, label="Training MSE")
plt.plot(val_losses, label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training and Validation Loss")
plt.legend()

plt.savefig(f"{workspace}/models/model_outputs/unet10_std_training_validation_loss_{timestamp}.png", dpi=300)
plt.close()

plt.figure(figsize=(8,4))
plt.plot(train_mae_list, label="Train MAE")
plt.plot(val_mae_list, label="Val MAE")
plt.plot(train_mse_list, label="Train MSE", linestyle="--")
plt.plot(val_mse_list, label="Val MSE",   linestyle="--")
plt.yscale("log")            # optional if the curves differ a lot
plt.xlabel("Epoch"); plt.ylabel("Error")
plt.title("Learning Curves – MAE & MSE")
plt.legend(); plt.tight_layout()
plt.savefig(f"{workspace}/models/model_outputs/unet10_mae_mse_{timestamp}.png", dpi=300)
plt.close()

writer.close()

model.eval()

# Inference loop -- recover predictions by adding back mean
predictions, targets, dyn_means = [], [], []

with torch.no_grad():
    for s, d, t in validation_loader:          # d is (B,2) after your Z‑score
        s = s.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        with autocast():
            pred_anom = model(s, d)

        # move to CPU
        pred_anom = pred_anom.squeeze(1).cpu().numpy() # (B,H,W)
        targ_anom = t.squeeze(1).cpu().numpy() # (B,H,W)

        # Need to fix this part
        dyn_means.append(validation_loader.last_raw_sm_mean.cpu().numpy())  # (B,)

        predictions.append(pred_anom)
        targets.append(targ_anom)

# concat
pred_anom = np.concatenate(predictions, axis=0) # (N,H,W)
targ_anom = np.concatenate(targets, axis=0) # (N,H,W)
sm_means = np.concatenate(dyn_means, axis=0) # (N,)

# broadcast means back to maps
sm_means_map = sm_means[:, None, None] # (N,1,1)
pred_full = pred_anom + sm_means_map
targ_full = targ_anom + sm_means_map

np.save(f"{workspace}/models/model_outputs/unet10_predictions_{timestamp}.npy", predictions)
np.save(f"{workspace}/models/model_outputs/unet10_targets_{timestamp}.npy",     targets)
print("Saved predictions and targets.")
