import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_target(x, target_size):
    """
    Pad tensor `x` (N,C,H,W) so its spatial size matches `target_size=(H_t,W_t)`.
    """
    _, _, h, w = x.shape
    target_h, target_w = target_size
    pad_h = target_h - h
    pad_w = target_w - w
    pad_top,  pad_bottom = pad_h // 2,  pad_h - pad_h // 2
    pad_left, pad_right  = pad_w // 2,  pad_w - pad_w // 2
    return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

class UNetFusionExplicit(nn.Module):
    """
    7 encoder blocks, 6 decoder blocks, skip connections, and drop out.
    """
    def __init__(
        self,
        in_channels_static: int = 33,
        dyn_channels: int = 2,
        out_channels: int = 1,
        inject_level: int = -2, 
    ):
        super().__init__()

        #  Encoder ― sizes: 32-64-128-256-512-512-1024
        self.inject_level = inject_level % 7
        self.dyn_channels = dyn_channels

        # block 0: 33 (in_channels) to 32
        self.d0_conv1 = nn.Conv2d(in_channels_static, 32, 3, padding=1)
        self.d0_drop1 = nn.Dropout2d(0.2)
        self.d0_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool0    = nn.MaxPool2d(2, 2)

        # Function to choose correct number of channels if it's time to inject the dynamic
        def in_ch(prev_out, idx):          # idx = current block index
            if idx - 1 == self.inject_level:
                return prev_out + dyn_channels
            return prev_out

        # block 1: 32 to 64
        self.d1_conv1 = nn.Conv2d(in_ch(32, 1), 64, 3, padding=1)
        self.d1_drop1 = nn.Dropout2d(0.2)
        self.d1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1    = nn.MaxPool2d(2, 2)

        # block 2: 64 to 128
        self.d2_conv1 = nn.Conv2d(in_ch(64, 2), 128, 3, padding=1)
        self.d2_drop1 = nn.Dropout2d(0.2)
        self.d2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2    = nn.MaxPool2d(2, 2)

        # block 3: 128 to 256
        self.d3_conv1 = nn.Conv2d(in_ch(128, 3), 256, 3, padding=1)
        self.d3_drop1 = nn.Dropout2d(0.2)
        self.d3_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3    = nn.MaxPool2d(2, 2)

        # block 4: 256 to 512
        self.d4_conv1 = nn.Conv2d(in_ch(256, 4), 512, 3, padding=1)
        self.d4_drop1 = nn.Dropout2d(0.2)
        self.d4_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4    = nn.MaxPool2d(2, 2)

        # block 5: 512 to 512
        self.d5_conv1 = nn.Conv2d(in_ch(512, 5), 512, 3, padding=1)
        self.d5_drop1 = nn.Dropout2d(0.2)
        self.d5_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5    = nn.MaxPool2d(2, 2)

        # block 6: 512 to 1024
        self.d6_conv1 = nn.Conv2d(in_ch(512, 6), 1024, 3, padding=1)
        self.d6_drop1 = nn.Dropout2d(0.2)
        self.d6_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.pool6    = nn.MaxPool2d(2, 2)   # will produce 2×2 for 360-pixel inputs

        #  Decoder ― up-convs followed by two plain convs (ReLU + Dropout + ReLU)

        # up-5: 1024 to 512, skip = 512
        self.up5_convT = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.u5_conv1  = nn.Conv2d(1024, 512, 3, padding=1)
        self.u5_drop1  = nn.Dropout2d(0.2)
        self.u5_conv2  = nn.Conv2d(512, 512, 3, padding=1)

        # up-4: 512 to 512, skip = 512
        self.up4_convT = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.u4_conv1  = nn.Conv2d(1024, 512, 3, padding=1)
        self.u4_drop1  = nn.Dropout2d(0.2)
        self.u4_conv2  = nn.Conv2d(512, 512, 3, padding=1)

        # up-3: 512 to 256, skip = 256
        self.up3_convT = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.u3_conv1  = nn.Conv2d(512, 256, 3, padding=1)
        self.u3_drop1  = nn.Dropout2d(0.2)
        self.u3_conv2  = nn.Conv2d(256, 256, 3, padding=1)

        # up-2: 256 to 128, skip = 128
        self.up2_convT = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u2_conv1  = nn.Conv2d(256, 128, 3, padding=1)
        self.u2_drop1  = nn.Dropout2d(0.2)
        self.u2_conv2  = nn.Conv2d(128, 128, 3, padding=1)

        # up-1: 128 to 64, skip = 64
        self.up1_convT = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u1_conv1  = nn.Conv2d(128, 64, 3, padding=1)
        self.u1_drop1  = nn.Dropout2d(0.2)
        self.u1_conv2  = nn.Conv2d(64, 64, 3, padding=1)

        # up-0: 64 to 32, skip = 32
        self.up0_convT = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.u0_conv1  = nn.Conv2d(64, 32, 3, padding=1)
        self.u0_drop1  = nn.Dropout2d(0.2)
        self.u0_conv2  = nn.Conv2d(32, 32, 3, padding=1)

        # final 1×1
        
        self.out_conv = nn.Conv2d(32, out_channels, 1)

    # Define forward pass
    def forward(self, x_static, dyn_vec):
        """
        x_static: (B, 33, 360, 360)
        dyn_vec: (B, 2)         – injected after encoder block `inject_level`
        """
        # Encoder
        # block 0: 33 to 32
        x = F.relu(self.d0_conv1(x_static))
        x = self.d0_drop1(x)
        x = F.relu(self.d0_conv2(x))
        skip0 = x # 32 -- save the skip connections
        x = self.pool0(x) if x.shape[-2] > 1 and x.shape[-1] > 1 else x
        if self.inject_level == 0:
            dyn = dyn_vec[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, dyn], dim=1)

        # block 1: 32 to 64
        x = F.relu(self.d1_conv1(x))
        x = self.d1_drop1(x)
        x = F.relu(self.d1_conv2(x))
        skip1 = x # 64
        x = self.pool1(x)
        if self.inject_level == 1:
            dyn = dyn_vec[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, dyn], dim=1)

        # block 2: 64 to 128
        x = F.relu(self.d2_conv1(x))
        x = self.d2_drop1(x)
        x = F.relu(self.d2_conv2(x))
        skip2 = x # 128
        x = self.pool2(x)
        if self.inject_level == 2:
            dyn = dyn_vec[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, dyn], dim=1)

        # block 3: 128 to 256
        x = F.relu(self.d3_conv1(x))
        x = self.d3_drop1(x)
        x = F.relu(self.d3_conv2(x))
        skip3 = x # 256
        x = self.pool3(x)
        if self.inject_level == 3:
            dyn = dyn_vec[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, dyn], dim=1)

        # block 4: 256 to 512
        x = F.relu(self.d4_conv1(x))
        x = self.d4_drop1(x)
        x = F.relu(self.d4_conv2(x))
        skip4 = x # 512
        x = self.pool4(x)
        if self.inject_level == 4:
            dyn = dyn_vec[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, dyn], dim=1)

        # block 5: 512 to 512
        x = F.relu(self.d5_conv1(x))
        x = self.d5_drop1(x)
        x = F.relu(self.d5_conv2(x))
        skip5 = x # 512
        x = self.pool5(x)
        if self.inject_level == 5:
            dyn = dyn_vec[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, dyn], dim=1)

        # block 6: 512 to 1024
        x = F.relu(self.d6_conv1(x))
        x = self.d6_drop1(x)
        x = F.relu(self.d6_conv2(x))
        skip6 = x # not used

        # Decoder
        # up-5
        x = self.up5_convT(x) # 1024 to 512
        if x.size(2) != skip5.size(2) or x.size(3) != skip5.size(3):
            x = pad_to_target(x, skip5.shape[-2:])
        x = torch.cat([skip5, x], dim=1) # add 512
        x = F.relu(self.u5_conv1(x))
        x = self.u5_drop1(x)
        x = F.relu(self.u5_conv2(x))

        # up-4
        x = self.up4_convT(x) # 512 to 512
        if x.size(2) != skip4.size(2) or x.size(3) != skip4.size(3):
            x = pad_to_target(x, skip4.shape[-2:])
        x = torch.cat([skip4, x], dim=1) # add 512
        x = F.relu(self.u4_conv1(x))
        x = self.u4_drop1(x)
        x = F.relu(self.u4_conv2(x))

        # up-3
        x = self.up3_convT(x) # 512 to 256
        if x.size(2) != skip3.size(2) or x.size(3) != skip3.size(3):
            x = pad_to_target(x, skip3.shape[-2:])
        x = torch.cat([skip3, x], dim=1) # add 256
        x = F.relu(self.u3_conv1(x))
        x = self.u3_drop1(x)
        x = F.relu(self.u3_conv2(x))

        # up-2
        x = self.up2_convT(x) # 256 to 128
        if x.size(2) != skip2.size(2) or x.size(3) != skip2.size(3):
            x = pad_to_target(x, skip2.shape[-2:])
        x = torch.cat([skip2, x], dim=1) # add 128
        x = F.relu(self.u2_conv1(x))
        x = self.u2_drop1(x)
        x = F.relu(self.u2_conv2(x))

        # up-1
        x = self.up1_convT(x) # 128 to 64
        if x.size(2) != skip1.size(2) or x.size(3) != skip1.size(3):
            x = pad_to_target(x, skip1.shape[-2:])
        x = torch.cat([skip1, x], dim=1) # add 64
        x = F.relu(self.u1_conv1(x))
        x = self.u1_drop1(x)
        x = F.relu(self.u1_conv2(x))

        # up-0
        x = self.up0_convT(x) # 64 to 32
        if x.size(2) != skip0.size(2) or x.size(3) != skip0.size(3):
            x = pad_to_target(x, skip0.shape[-2:])
        x = torch.cat([skip0, x], dim=1) # add 32
        x = F.relu(self.u0_conv1(x))
        x = self.u0_drop1(x)
        x = F.relu(self.u0_conv2(x))

        # output
        return self.out_conv(x) # 32 to out_channels
