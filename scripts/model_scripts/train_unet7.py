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

from data_loader_unet7 import CombinedDataset
from unet7 import UNetFusionExplicit

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
                 'bd_60_100',
                 'clay_0_5',
                 'commercial',
                 'cropland',
                 'elevation',
                 'exurban_high',
                 'exurban_low',
                 'grazing',
                 'hb_0_5',
                 'industrial',
                 'institutional',
                 'ksat_0_5',
                 'mining_barren_land',
                 'n_0_5',
                 'natural_water',
                 'om_0_5',
                 'parks_golf_courses',
                 'pasture',
                 'ph_0_5',
                 'ph_60_100',
                 'recreation_conservation',
                 'reservoirs_canals',
                 'sand_0_5',
                 'silt_0_5',
                 'suburban',
                 'theta_r_0_5',
                 'theta_s_0_5',
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

optimizer = optim.Adam(model.parameters(), lr=3e-5)
loss_fn = nn.MSELoss()

def train_one_epoch(epoch, writer=None):
    model.train()
    total_loss = 0.0
    start_epoch_time = time.time()
    
    start = time.perf_counter()
    
    data_iter = iter(training_loader)
    for step in range(len(training_loader)):     
        start = time.perf_counter()
        static_maps, dyn_vec, target = next(data_iter)
        loader_ms = (time.perf_counter() - start) * 1e3  
        
        torch.cuda.synchronize()
        
        # For timing
        torch.cuda.synchronize()
        start_gpu = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            pred = model(static_maps, dyn_vec)                 
            loss = loss_fn(pred, target) 
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 
        
        torch.cuda.synchronize()
        gpu_t = time.perf_counter() - start_gpu
        
        if step < 20: # Print out for first 20 batches
            print(f"step {step:<3} loader {loader_ms:6.1f} ms  gpu {gpu_t*1e3:6.1f} ms")
        start = time.perf_counter()

        total_loss += loss.item()

    epoch_time = time.time() - start_epoch_time
    print(f"Epoch {epoch} finished in {epoch_time:.2f} seconds")

    # Average loss
    avg_loss = total_loss / len(training_loader)
    print(f"Epoch {epoch}, average train loss={avg_loss:.4f}")

    if writer is not None:
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Timing/train_epoch_seconds", epoch_time, epoch)

    return avg_loss

def validate_one_epoch(epoch, writer=None):
    model.eval()
    total_vloss = 0.0
    start_val_time = time.time()

    with torch.no_grad():
        data_iter = iter(validation_loader)
        for step in range(len(validation_loader)):
            s, d, t = next(data_iter)
            with autocast():
                pred  = model(s, d)
                vloss = loss_fn(pred, t)
            total_vloss += vloss.item()

    val_time = time.time() - start_val_time
    print(f"Validation for epoch {epoch} took {val_time:.2f} seconds")

    avg_vloss = total_vloss / len(validation_loader)
    print(f"Validation MSE: {avg_vloss:.4f}")

    if writer is not None:
        writer.add_scalar("Loss/val", avg_vloss, epoch)
        writer.add_scalar("Timing/val_epoch_seconds", val_time, epoch)

    return avg_vloss

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=f"{workspace}/models/runs/unet7_trainer_{timestamp}")

num_epochs = 100
best_vloss = float("inf") # set to max to start

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    print(f"--- EPOCH {epoch+1}/{num_epochs} ---")

    # Train
    avg_loss = train_one_epoch(epoch, writer=writer)
    train_losses.append(avg_loss)

    # Validate
    avg_vloss = validate_one_epoch(epoch, writer=writer)
    val_losses.append(avg_vloss)

    print(f"Train MSE: {avg_loss:.6f}, Val MSE: {avg_vloss:.6f}")

    # Save best model, keep track of best validation loss to determine if this epoch does better than prev best
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"{workspace}/models/model_outputs/unet7_std_model_{timestamp}_epoch{epoch}.pth"
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

plt.savefig(f"{workspace}/models/model_outputs/unet7_std_training_validation_loss_{timestamp}.png", dpi=300)
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

np.save(f"{workspace}/models/model_outputs/unet7_std_predictions_{timestamp}.npy", predictions)
np.save(f"{workspace}/models/model_outputs/unet7_std_targets_{timestamp}.npy",     targets)
print("Saved predictions and targets.")
