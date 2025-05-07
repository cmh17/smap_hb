import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_to_target(x, target_size):
    _, _, h, w = x.shape
    target_h, target_w = target_size
    pad_h = target_h - h
    pad_w = target_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')
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
        conv_out = self.conv_block(x) 
        if conv_out.size(-2) > 1 and conv_out.size(-1) > 1:
            downsampled = self.pool(conv_out)
        else:
            downsampled = conv_out 
        return downsampled, conv_out

class UpBlock(nn.Module):
    def __init__(self, in_channels_up, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels_up, out_channels,
                                         kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        
        # Use your existing pad_to_target function if there's a mismatch in spatial size
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = pad_to_target(x, (skip.size(2), skip.size(3)))
        
        # Now both tensors should have the same H×W
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x


class UNetFusion(nn.Module):
    def __init__(self, in_channels_static=86, in_channels_dynamic=2, out_channels=1):
        super().__init__()

        self.down_channels = [64, 128, 256, 512, 512, 512, 512, 512, 512]
        
        in_c = in_channels_static
        self.down_blocks = nn.ModuleList()
        for out_c in self.down_channels:
            self.down_blocks.append(DownBlock(in_c, out_c))
            in_c = out_c
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(self.down_channels[-1] + in_channels_dynamic, self.down_channels[-1], kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(self.down_channels))
        
        for i in range(len(reversed_channels) - 1):
            up_in   = reversed_channels[i]       # channels from the previous up block
            skip_c  = reversed_channels[i + 1]   # channels from the skip
            out_c   = reversed_channels[i + 1]   # final output
            self.up_blocks.append(UpBlock(up_in, skip_c, out_c))

        
        self.out_conv = nn.Conv2d(self.down_channels[0], out_channels, kernel_size=1)
    
    def forward(self, static_x, dynamic_x):

        skips = []
        x = static_x
        for down in self.down_blocks:
            x, skip = down(x)
            skips.append(skip)
#             print(f"DownBlock: x shape: {x.shape}, skip shape: {skip.shape}")
        
        x = torch.cat([x, dynamic_x], dim=1)
        x = self.fuse_conv(x)

        used_skips = skips[:-1] # discard last skip
        used_skips = used_skips[::-1]

        decoded = x
        for i in range(len(self.up_blocks)):
            skip = used_skips[i]
            decoded = self.up_blocks[i](decoded, skip)
#             print(f"UpBlock: decoded shape: {decoded.shape}, skip shape: {skip.shape}")

        out = self.out_conv(decoded)
        return out
