import torch
import torchvision

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, (1, 1)),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Hypernet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)

        self.decoder4 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 256)
        self.decoder2 = Decoder(128, 256)
        self.decoder1 = Decoder(64, 256)

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, (3, 3), padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 256, (3, 3), padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, num_classes, 3, padding=1),
            torch.nn.Upsample(scale_factor=4, mode='bilinear')
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        f4 = self.decoder4(x4)
        f3 = self.decoder3(x3)
        f2 = self.decoder2(x2)
        f1 = self.decoder1(x1)

        return self.classifier(torch.cat([
            torch.nn.functional.upsample_bilinear(f4, scale_factor=8),
            torch.nn.functional.upsample_bilinear(f3, scale_factor=4),
            torch.nn.functional.upsample_bilinear(f2, scale_factor=2),
            f1
        ], dim=1))
