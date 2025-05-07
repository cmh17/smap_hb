import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_to_target(x, target_size):
    """
    Function to pad the input to match the target 
    """
    _, _, h, w = x.shape
    target_h, target_w = target_size
    pad_h = target_h - h
    pad_w = target_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top # pad asymmetrically if needed
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same') # pads if needed
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding='same')
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_block = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.conv_block(x) 
        # Downsample if shape > 1x1
        if out.shape[-2] > 1 and out.shape[-1] > 1:
            out_down = self.pool(out)
        else: # Do nothing if already 1x1
            out_down = out
        return out_down, out  # downsampled, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels_up, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels_up, out_channels,
                                         kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        
        # Pad if needed so that x matches skip size
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = pad_to_target(x, (skip.size(2), skip.size(3)))
        
        # Concatenate
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x

class UNetFusion(nn.Module):
    def __init__(self, in_channels=35, out_channels=1):
        """
        Simple UNet that expects 'in_channels' (currently default 35) total input channels.
        """
        super().__init__()

        # Try adding more layers vs unet3 -- this takes it down to 2x2
        #self.down_channels = [32, 64, 128, 256, 512, 512, 1024]
        self.down_channels = [32, 64, 128, 256, 512, 512, 1024]

        # Create DownBlocks
        in_c = in_channels
        self.down_blocks = nn.ModuleList()
        for out_c in self.down_channels:
            self.down_blocks.append(DownBlock(in_c, out_c))
            in_c = out_c
            
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(self.down_channels))
        for i in range(len(reversed_channels) - 1):
            up_in  = reversed_channels[i]
            skip_c = reversed_channels[i + 1]
            out_c  = reversed_channels[i + 1]
            self.up_blocks.append(UpBlock(up_in, skip_c, out_c))

        self.out_conv = nn.Conv2d(self.down_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Down path
        skips = []
        for idx, down in enumerate(self.down_blocks):
            x, skip = down(x)
            print(f"DownBlock {idx}: x.shape={x.shape}")  # <--- debug
            skips.append(skip)

        decoded = x
        n = len(self.up_blocks)
        for i in range(n):
            skip = skips[-(i+2)]
            decoded = self.up_blocks[i](decoded, skip)

        # Output
        out = self.out_conv(decoded)
        return out
