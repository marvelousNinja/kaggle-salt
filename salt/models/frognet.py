import torch
import torchvision

class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)

class SCSEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_gate = torch.nn.Sequential(
            GlobalAvgPool2d(),
            torch.nn.Linear(in_channels, in_channels // reduction),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_channels // reduction, in_channels),
            torch.nn.Sigmoid()
        )

        self.spatial_gate = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 1, 1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.spatial_gate(x) * x + self.channel_gate(x).view(x.shape[0], -1, 1, 1) * x

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            SCSEBlock(out_channels)
        )

    def forward(self, x):
        return self.layers(x)

class Frognet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)

        self.center = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )

        self.encoders = torch.nn.ModuleList([
            torch.nn.Sequential(
                self.resnet.conv1,
                self.resnet.bn1,
                self.resnet.relu
                # Maxpool removed intentionally
            ),
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ])

        self.decoders = torch.nn.ModuleList([
            Decoder(256 + 512, 512, 64),
            Decoder(64 + 256, 256, 64),
            Decoder(64 + 128, 128, 64),
            Decoder(64 + 64, 64, 64),
            Decoder(64, 32, 64)
        ])

        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(320, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        x = self.encoders[0](x)
        x1 = self.encoders[1](x)
        x2 = self.encoders[2](x1)
        x3 = self.encoders[3](x2)
        x4 = self.encoders[4](x3)

        f = self.center(x4)

        d5 = self.decoders[0](torch.cat([f, x4], dim=1))
        d4 = self.decoders[1](torch.cat([d5, x3], dim=1))
        d3 = self.decoders[2](torch.cat([d4, x2], dim=1))
        d2 = self.decoders[3](torch.cat([d3, x1], dim=1))
        d1 = self.decoders[4](d2)

        f = torch.cat([
            d1,
            torch.nn.functional.upsample_bilinear(d2, scale_factor=2),
            torch.nn.functional.upsample_bilinear(d3, scale_factor=4),
            torch.nn.functional.upsample_bilinear(d4, scale_factor=8),
            torch.nn.functional.upsample_bilinear(d5, scale_factor=16)
        ], dim=1)

        f = self.dropout(f)
        return self.classifier(f)

