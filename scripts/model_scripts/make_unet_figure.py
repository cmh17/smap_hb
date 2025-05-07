# Install torchview and torchinfo
# pip install torchview torchinfo

import torch
from torchview import draw_graph          # picture
from torchinfo import summary             # table
from unet7 import UNetFusionExplicit      # unet class I'm currently using

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = UNetFusionExplicit(in_channels_static=33, out_channels=1).to(device) # current num channels is used here

B, H, W = 2, 360, 360 # dummy inputs
x_static = torch.randn(B, 33, H, W, device=device)
dyn_vec  = torch.randn(B, 2, device=device)

# picture
graph = draw_graph(model,
                   input_data=(x_static, dyn_vec),
                   graph_dir=".",          # where to save
                   save_graph=True,
                   filename="unet7_arch",  # creates unet7_arch.png
                   expand_nested=True,     # show every Conv layer
                   roll=True)              # arrange top‑down
print("PNG saved to unet7_arch.png")

# 3) console output
print(summary(model,
              input_data=(x_static, dyn_vec),
              col_names=("input_size", "output_size", "num_params"),
              row_settings=("depth", "var_names")))
