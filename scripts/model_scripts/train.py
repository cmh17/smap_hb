import os
from torch.utils.data import DataLoader, Subset
from data_loader import CombinedDataset
from unet import UNetFusion
import torch
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Did this bc of local error... might cause problems though

workspace = os.path.dirname(os.getcwd())
root_dir = f"{workspace}/data/combined_output/tiles2"

dataset = CombinedDataset(root_dir, time_step=1) # use all for now
                            # listofstatic=["elevation", "silt_0_5", "sand_0_5", 
                            #               "clay_0_5", "urban_low", "urban_high", 
                            #               "commercial", "industrial"])

print("Dataset size:", len(dataset))

# Use a subset to only load the first 4 tiles
# subset_indices = list(range(4))
# small_dataset = Subset(dataset, subset_indices)

# loader = DataLoader(small_dataset, batch_size=1, shuffle=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)


print("DataLoader size:", len(loader))

import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetFusion(in_channels_static=86, in_channels_dynamic=2, out_channels=1)
model.to(device)
# model.eval() 
# print("Model loaded on device:", next(model.parameters()).device)

# with torch.no_grad():
#     for static_data, dynamic_data, target_data in loader:
#         # Move data to device
#         static_data = static_data.to(device)
#         dynamic_data = dynamic_data.to(device)
#         target_data = target_data.to(device)

#         dynamic_data = dynamic_data.squeeze(dim=2) # Remove singleton time dim
#         target_data = target_data.squeeze(dim=2)
        
#         # Get model prediction
#         pred = model(static_data, dynamic_data)
#         print("Prediction shape:", pred.shape)
#         print("Target shape:", target_data.shape)
        
#         pred_np = pred.cpu().squeeze().numpy()  # squeeze to remove batch and channel dims
#         target_np = target_data.cpu().squeeze().numpy()
        
#         pred_mean = pred_np.mean()
#         pred_std = pred_np.std()
#         targ_mean = target_np.mean()
#         targ_std = target_np.std()

#         print("Prediction min/max:", pred_np.min(), pred_np.max())
#         print("Prediction mean/std:", pred_mean, pred_std)
#         print("Target min/max:", target_np.min(), target_np.max())
#         print("Target mean/std:", targ_mean, targ_std)

#         mse = torch.mean((pred - target_data)**2)
#         print("MSE (PyTorch):", mse.item()) 


#         # break

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()
num_epochs = 10

epoch_loss = 0.0

loss_list  = []

for epoch in range(num_epochs):
    for static_data, dynamic_data, target_data in loader:
        static_data = static_data.to(device)
        dynamic_data = dynamic_data.to(device)
        target_data = target_data.to(device)
        
        dynamic_data = dynamic_data.squeeze(dim=2)
        target_data = target_data.squeeze(dim=2)
        
        # Forward pass
        pred = model(static_data, dynamic_data)
        
        # Compute loss
        loss = criterion(pred, target_data)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        loss_list.append(loss.item())
    
    print(f"Epoch {epoch}, loss={loss.item():.4f}")
    epoch_loss += loss.item()

print("Training complete!")

# Save the model
torch.save(model.state_dict(), "model.pth")
print("Model saved!")

print(loss_list)


