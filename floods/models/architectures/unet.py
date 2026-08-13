import torch
import torch.nn as nn
import torch.nn.functional as F

from floods.models.architectures.base_model import BaseModel


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)
        if interpolate:
            # Interpolate to align skip-connection dimensions.
            x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)), mode="bilinear", align_corners=True)
        else:
            # Pad when incoming feature maps have different spatial dimensions.
            diff_y = x_copy.size()[2] - x.size()[2]
            diff_x = x_copy.size()[3] - x.size()[3]
            x = F.pad(x, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))
        # Concatenate the upsampled tensor with the matching skip connection.
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class UNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(UNet, self).__init__()
        self.down1 = EncoderBlock(in_channels, 64)
        self.down2 = EncoderBlock(64, 128)
        self.down3 = EncoderBlock(128, 256)
        self.down4 = EncoderBlock(256, 512)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.up1 = DecoderBlock(1024, 512)
        self.up2 = DecoderBlock(512, 256)
        self.up3 = DecoderBlock(256, 128)
        self.up4 = DecoderBlock(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()
        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x = self.middle_conv(x)
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.final_conv(x)
        return x, None

    def get_backbone_params(self):
        # This implementation has no separate backbone parameter group.
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
