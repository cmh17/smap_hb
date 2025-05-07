import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt

from data_loader import CombinedDataset
from unet import UNetFusion

# -------------------------------------------------------------------------
# 1) Set up paths and datasets

workspace = os.path.dirname(os.getcwd())

# Example: use time_step=1 for training, time_step=3 for validation
training_set = CombinedDataset(root_dir, time_step=1)
validation_set = CombinedDataset(root_dir, time_step=3)

print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))

# Create separate DataLoaders
batch_size = 4
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

# -------------------------------------------------------------------------
# 2) Prepare model, optimizer, etc.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetFusion(in_channels_static=86, in_channels_dynamic=2, out_channels=1)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# -------------------------------------------------------------------------
# 3) Define a function to do one epoch of training

def train_one_epoch(epoch_index, writer=None):
    model.train()
    total_loss = 0.0

    for i, (static_data, dynamic_data, target_data) in enumerate(training_loader):
        # Move data to device
        static_data = static_data.to(device)
        dynamic_data = dynamic_data.to(device)
        target_data = target_data.to(device)
        
        # Forward pass
        pred = model(static_data, dynamic_data)
        loss = loss_fn(pred, target_data)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(training_loader)
    
    # Optionally log to TensorBoard
    if writer is not None:
        writer.add_scalar('Loss/train', avg_loss, epoch_index)
    
    return avg_loss


# -------------------------------------------------------------------------
# 4) Define a function to compute validation loss

def validate_one_epoch(epoch_index, writer=None):
    model.eval()
    total_vloss = 0.0

    with torch.no_grad():
        for i, (static_data, dynamic_data, target_data) in enumerate(validation_loader):
            static_data = static_data.to(device)
            dynamic_data = dynamic_data.to(device)
            target_data = target_data.to(device)

            pred = model(static_data, dynamic_data)
            vloss = loss_fn(pred, target_data)
            total_vloss += vloss.item()

    avg_vloss = total_vloss / len(validation_loader)
    
    # TensorBoard
    if writer is not None:
        writer.add_scalar('Loss/val', avg_vloss, epoch_index)
    
    return avg_vloss


# -------------------------------------------------------------------------
# 5) Main training loop

# Set up TensorBoard writer if you like
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(log_dir=f'runs/unet_trainer_{timestamp}')

num_epochs = 10
best_vloss = float('inf')

# For plotting after training
train_losses = []
val_losses   = []

for epoch in range(num_epochs):
    print(f"--- EPOCH {epoch+1}/{num_epochs} ---")
    
    # Train for one epoch
    avg_loss = train_one_epoch(epoch, writer=writer)
    train_losses.append(avg_loss)
    
    # Validate for one epoch
    avg_vloss = validate_one_epoch(epoch, writer=writer)
    val_losses.append(avg_vloss)

    print(f"Train MSE: {avg_loss:.6f}, Val MSE: {avg_vloss:.6f}")

    # Save the model if it’s the best yet
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"model_{timestamp}_epoch{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  -> Saved new best model to {model_path}")

# -------------------------------------------------------------------------
# 6) Plot the training/validation loss curves

plt.figure()
plt.plot(train_losses, label='Training MSE')
plt.plot(val_losses, label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training and Validation Loss')
plt.legend()

# Save the figure to a file
plt.savefig('training_validation_loss1.png', dpi=300)

# Close the TensorBoard writer 
writer.close()
