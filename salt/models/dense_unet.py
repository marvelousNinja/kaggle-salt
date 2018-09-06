import torch
import torchvision

class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        out_per_block = out_channels // 4
        channel_spec = [
            (in_channels, out_per_block),
            (in_channels + out_per_block, out_per_block),
            (out_per_block * 2, out_per_block),
            (out_per_block * 2, out_per_block)
        ]

        self.layers = torch.nn.ModuleList()
        for (in_ch, out_ch) in channel_spec:
            self.layers.append(torch.nn.Sequential(
                torch.nn.BatchNorm2d(in_ch),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
                # Dropout ?
            ))

    def forward(self, x):
        x0 = self.layers[0](x)
        x1 = self.layers[1](torch.cat([x0, x], dim=1))
        x2 = self.layers[2](torch.cat([x1, x0], dim=1))
        x3 = self.layers[3](torch.cat([x2, x1], dim=1))
        return torch.cat([x0, x1, x2, x3], dim=1)

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            DenseBlock(in_channels, out_channels),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.layers(x)

class DenseUnet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.densenet = torchvision.models.densenet121(pretrained=True)
        self.decoder1 = Decoder(1024, 256)
        self.decoder2 = Decoder(1280, 256)
        self.decoder3 = Decoder(768, 256)
        self.decoder4 = Decoder(512, 256)
        self.decoder5 = Decoder(320, 64)
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        x = self.densenet.features[0](x)
        x = self.densenet.features[1](x)
        x0 = self.densenet.features[2](x) # torch.Size([8, 64, 64, 64])
        x1 = self.densenet.features[4](self.densenet.features[3](x0)) # torch.Size([8, 256, 32, 32])
        x2 = self.densenet.features[6](self.densenet.features[5](x1)) # torch.Size([8, 512, 16, 16])
        x3 = self.densenet.features[8](self.densenet.features[7](x2)) # torch.Size([8, 1024, 8, 8])
        x4 = self.densenet.features[10](self.densenet.features[9](x3)) # torch.Size([8, 1024, 4, 4])
        x = self.decoder1(x4)
        x = self.decoder2(torch.cat([x, x3], dim=1))
        x = self.decoder3(torch.cat([x, x2], dim=1))
        x = self.decoder4(torch.cat([x, x1], dim=1))
        x = self.decoder5(torch.cat([x, x0], dim=1))
        return self.classifier(x)
