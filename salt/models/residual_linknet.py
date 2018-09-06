import torch
import torchvision

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upscaler = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.smoother = torch.nn.Sequential(
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

        self.projector = torch.nn.Conv2d(in_channels, out_channels, 1)

        self.extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.extractor(x) + self.projector(x)
        x = self.upscaler(x)
        x = self.smoother(x) + x
        return x

class Linknet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.decoder1 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder3 = Decoder(128, 64)
        self.decoder4 = Decoder(64, 64)
        self.decoder5 = Decoder(64, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x0 = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x0)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        x = self.decoder1(x4) + x3
        x = self.decoder2(x) + x2
        x = self.decoder3(x) + x1
        x = self.decoder4(x)
        return self.decoder5(x)
