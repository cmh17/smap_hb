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

from data_loader_unet8 import CombinedDataset # Not changing these, just the loss function
from unet8 import UNetFusionExplicit

from torch.cuda.amp import autocast # Try using half precision when safe
from torch.cuda.amp import GradScaler
scaler = GradScaler() 

import random, numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)

# import torch.multiprocessing as mp # Try to make it so I can use multiple workers... no luck with this yet, memory errors
# mp.set_sharing_strategy("file_descriptor") 
# print("Sharing strategy:", mp.get_sharing_strategy())

workspace = os.path.dirname(os.path.dirname(os.getcwd()))
root_dir = f"{workspace}/data/combined_output/tiles2"
os.makedirs(os.path.join(workspace, "models", "model_outputs"), exist_ok=True)

loading_start_time = time.time()

ds = xr.open_dataset(os.path.join(workspace, "data", "combined_output", "tiles2", "tile_0_0", "dynamic.nc"))
time_inds = list(range(0, (len(ds.time.values))))

years = ds.time.values.astype("datetime64[Y]").astype(int) + 1970

# Take time slices from 2016-2018, stepping by 10 (by days, but don't have all days)
# time_inds = list(range(list(years).index(2016), list(years).index(2018), 10))

train_time_inds = list(range(list(years).index(2016), list(years).index(2017), 10))
test_time_inds = list(range(list(years).index(2017), list(years).index(2018), 10))

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
               'wetlands']

training_set = CombinedDataset(root_dir, time_inds=train_time_inds, static_vars=static_vars,
                              static_stats_file=os.path.join(workspace, "data", "combined_output", "static_norm_stats.json"),
                              dynamic_stats_file=os.path.join(workspace, "data", "combined_output", "dynamic_norm_stats.json"),
                              cache_static=True,cache_device="cuda")
validation_set = CombinedDataset(root_dir, time_inds=test_time_inds, static_vars=static_vars,
                                 static_stats_file=os.path.join(workspace, "data", "combined_output", "static_norm_stats.json"),
                                 dynamic_stats_file=os.path.join(workspace, "data", "combined_output", "dynamic_norm_stats.json"),
                                 cache_static=True,cache_device="cuda")

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
model = UNetFusionExplicit(in_channels_static=33, out_channels=1).to(device) # 33 instead of 35 bc don't add dynamic immediately
if torch.cuda.device_count() > 1: # Try to use multiple GPUs
     model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Try hybrid loss function
mae_fn = torch.nn.L1Loss()
mse_fn = torch.nn.MSELoss()
alpha  = 0.6

def combined_loss(pred, target):
    return alpha * mse_fn(pred, target) + (1 - alpha) * mae_fn(pred, target)

loss_fn = combined_loss

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
writer = SummaryWriter(log_dir=f"{workspace}/models/runs/unet8_trainer_{timestamp}")

num_epochs = 150
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
        model_path = f"{workspace}/models/model_outputs/unet8_std_model_{timestamp}_epoch{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved new best model to {model_path}")

end_time = time.time()
print("Time to load, train, and validate: ", end_time - loading_start_time)

plt.figure()
plt.plot(train_losses, label="Training Mixed Loss")
plt.plot(val_losses, label="Validation Mixed Loss")
plt.xlabel("Epoch")
plt.ylabel("Mixed Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.savefig(f"{workspace}/models/model_outputs/unet8_std_training_validation_loss_{timestamp}.png", dpi=300)
plt.close()

plt.figure(figsize=(8,4))
plt.plot(train_mae_list, label="Train MAE")
plt.plot(val_mae_list,   label="Val MAE")
plt.plot(train_mse_list, label="Train MSE", linestyle="--")
plt.plot(val_mse_list,   label="Val MSE",   linestyle="--")
plt.yscale("log")            # optional if the curves differ a lot
plt.xlabel("Epoch"); plt.ylabel("Error")
plt.title("Learning Curves – MAE & MSE")
plt.legend(); plt.tight_layout()
plt.savefig(f"{workspace}/models/model_outputs/unet8_mae_mse_{timestamp}.png", dpi=300)
plt.close()

writer.close()

model.eval()
predictions, targets = [], []

with torch.no_grad():
    start_predict_time = time.time()
    for s, d, t in validation_loader:

        # forward pass with AMP
        with autocast():
            pred = model(s, d)        # (B, 1, H, W)

        # store to lists
        predictions.append(pred.squeeze(1).cpu().numpy())  # (B, H, W)
        targets.append(t.squeeze(1).cpu().numpy())         # (B, H, W)

    elapsed = time.time() - start_predict_time
    print(f"Inference on validation set took {elapsed:.2f} seconds")

predictions = np.concatenate(predictions, axis=0)
targets     = np.concatenate(targets,     axis=0)

np.save(f"{workspace}/models/model_outputs/unet8_std_predictions_{timestamp}.npy", predictions)
np.save(f"{workspace}/models/model_outputs/unet8_std_targets_{timestamp}.npy",     targets)
print("Saved predictions and targets.")
