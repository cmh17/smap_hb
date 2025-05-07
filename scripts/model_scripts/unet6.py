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
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2))
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
    def __init__(self,
                 in_channels_static=33,
                 dyn_channels=2,
                 out_channels=1,
                 inject_level=-2):         
        """
        inject_level < 0  counts from the end of down_blocks,
        so -1 == last block, -2 == penultimate, etc.
        """
        super().__init__()
        self.inject_level = inject_level % 7
        self.dyn_channels = dyn_channels

        self.down_channels = [32, 64, 128, 256, 512, 512, 1024]

        self.down_blocks = nn.ModuleList()
        in_c = in_channels_static
        for i, out_c in enumerate(self.down_channels):
            if i == self.inject_level + 1: # after adding dynamic, increase number of channels by adding dynamic
                in_c += dyn_channels
            self.down_blocks.append(DownBlock(in_c, out_c))
            in_c = out_c

        self.up_blocks = nn.ModuleList()
        rev = list(reversed(self.down_channels))
        for i in range(len(rev) - 1):
            self.up_blocks.append(
                UpBlock(rev[i], rev[i + 1], rev[i + 1])
            )

        self.out_conv = nn.Conv2d(self.down_channels[0], out_channels, 1)

    def forward(self, x_static, dyn_vec):
        """
        x_static : (B,33,360,360)
        dyn_vec  : (B,2)
        """
        skips, x = [], x_static
        for i, down in enumerate(self.down_blocks):
            x, skip = down(x)   # out_down, out_skip

            if i == self.inject_level:
                # dyn_vec ->  (B,2,1,1) -> repeat to 5×5
                dyn = dyn_vec[:, :, None, None].expand(-1, -1, x.size(-2), x.size(-1))
                x   = torch.cat([x, dyn], dim=1)      # concatenate to fuse the data

            skips.append(skip)

        for i, up in enumerate(self.up_blocks):
            x = up(x, skips[-(i + 2)])

        return self.out_conv(x)

