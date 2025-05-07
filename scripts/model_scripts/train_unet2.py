import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import xarray as xr
import time  # <-- We'll use this for measuring elapsed times

from data_loader_unet2 import CombinedDataset
from unet2 import UNetFusion

#############################################################################
# 0) Setup
# ############################################################################

workspace = os.path.dirname(os.getcwd())
root_dir = f"{workspace}/data/combined_output/tiles2"

loading_start_time = time.time()

ds = xr.open_dataset(os.path.join(workspace, "data", "combined_output", "tiles2", "tile_0_0", "dynamic.nc"))
time_inds = list(range(0, (len(ds.time.values))))

years = ds.time.values.astype("datetime64[Y]").astype(int) + 1970

# Take time slices from 2016-2018, stepping by 10
# time_inds = list(range(list(years).index(2016), list(years).index(2018), 10))

train_time_inds = list(range(list(years).index(2016), list(years).index(2017), 10))
test_time_inds = list(range(list(years).index(2017), list(years).index(2018), 10))

# train_tiles = []
# test_tiles = []

# for i in range(10):
#     for j in range(10):
#         new_tile = f"tile_{i}_{j}"
#         if (i + j) % 2 == 0:
#             train_tiles.append(new_tile)
#         else:
#             test_tiles.append(new_tile)

training_set = CombinedDataset(root_dir, time_inds=train_time_inds)
validation_set = CombinedDataset(root_dir, time_inds=test_time_inds)

print("Training set has {} instances".format(len(training_set)))
print("Validation set has {} instances".format(len(validation_set)))

batch_size = 4
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

loading_end_time = time.time()

loading_time = loading_end_time - loading_start_time
print(f"Loading data took {loading_time:.2f} seconds")

#############################################################################
# 1) Model / Optimizer Setup
# ############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetFusion(in_channels=88, out_channels=1).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

#############################################################################
# 2) Training & Validation Helpers
# ############################################################################

def train_one_epoch(epoch, writer=None):
    model.train()
    total_loss = 0.0
    start_epoch_time = time.time()

    for i, (predictor_data, target_data) in enumerate(training_loader):
        # Move data to device
        predictor_data = predictor_data.to(device)
        target_data    = target_data.to(device)

        # Forward pass (single input)
        pred = model(predictor_data)

        # Compute loss
        loss = loss_fn(pred, target_data)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        for i, (predictor_data, target_data) in enumerate(validation_loader):
            # Move data to device
            predictor_data = predictor_data.to(device)
            target_data    = target_data.to(device)

            # Single‐input forward
            pred = model(predictor_data)

            # Compute val loss
            vloss = loss_fn(pred, target_data)
            total_vloss += vloss.item()

    val_time = time.time() - start_val_time
    print(f"Validation for epoch {epoch} took {val_time:.2f} seconds")

    avg_vloss = total_vloss / len(validation_loader)
    print(f"Validation MSE: {avg_vloss:.4f}")

    if writer is not None:
        writer.add_scalar("Loss/val", avg_vloss, epoch)
        writer.add_scalar("Timing/val_epoch_seconds", val_time, epoch)

    return avg_vloss

#############################################################################
# 3) Main Training Loop
# ############################################################################

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=f"runs/unet_trainer_{timestamp}")

num_epochs = 20
best_vloss = float("inf")

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

    # Save best model
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"unet2_model_{timestamp}_epoch{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  -> Saved new best model to {model_path}")

plt.figure()
plt.plot(train_losses, label="Training MSE")
plt.plot(val_losses, label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training and Validation Loss")
plt.legend()

plt.savefig(f"unet2_training_validation_loss_{timestamp}.png", dpi=300)

writer.close()

#############################################################################
# 4) Collect predictions
# ############################################################################

import numpy as np

model.eval()
predictions = []
targets = []

with torch.no_grad():
    start_predict_time = time.time()
    for predictor_data, target_data in validation_loader:
        predictor_data = predictor_data.to(device)
        target_data = target_data.to(device)

        pred = model(predictor_data)
        
        pred_np   = pred.cpu().numpy()
        target_np = target_data.cpu().numpy()

        predictions.append(pred_np)
        targets.append(target_np)
    end_predict_time = time.time()

    print(f"Inference on validation set took {end_predict_time - start_predict_time:.2f} seconds")

predictions = np.concatenate(predictions, axis=0)
targets     = np.concatenate(targets,     axis=0)

np.save(f"unet2_predictions_{timestamp}.npy", predictions)
np.save(f"unet2_targets_{timestamp}.npy",     targets)
print("Saved predictions and targets.")
