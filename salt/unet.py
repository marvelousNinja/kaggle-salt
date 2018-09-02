import torch
import torchvision

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels, middle_channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(middle_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(middle_channels, out_channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Unet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.encoders = [
            Encoder(1, 64, 3, 1),
            Encoder(64, 128, 3, 1),
            Encoder(128, 256, 3, 1),
            Encoder(256, 512, 3, 1)
        ]

        self.decoders = [
            Decoder(512, 1024, 512, 3, 1),
            Decoder(1024, 512, 256, 3, 1),
            Decoder(512, 256, 128, 3, 1),
            Decoder(256, 128, 64, 3, 1)
        ]

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        x1 = self.encoders[0](x)
        x2 = self.encoders[1](self.maxpool(x1))
        x3 = self.encoders[2](self.maxpool(x2))
        x4 = self.encoders[3](self.maxpool(x3))
        d4 = self.decoders[0](self.maxpool(x4))
        d3 = self.decoders[1](torch.cat([d4, x4], dim=1))
        d2 = self.decoders[2](torch.cat([d3, x3], dim=1))
        d1 = self.decoders[3](torch.cat([d2, x2], dim=1))
        return self.final(torch.cat([d1, x1], dim=1))

class UnetVgg16(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vgg = torchvision.models.vgg16(pretrained=False).features
        self.maxpool = torch.nn.MaxPool2d(2)
        self.encoders = [
            torch.nn.Sequential(self.vgg[0:4]),   # 64
            torch.nn.Sequential(self.vgg[5:9]),   # 128
            torch.nn.Sequential(self.vgg[10:16]), # 256
            torch.nn.Sequential(self.vgg[17:23]), # 512
            torch.nn.Sequential(self.vgg[24:30])  # 512
        ]

        self.decoders = [
            Decoder(512, 512, 256, 3, 1),         # x5 (512)
            Decoder(512 + 256, 512, 256, 3, 1),   # x5 (512) + d5 (256)
            Decoder(512 + 256, 512, 256, 3, 1),   # x4 (512) + d4 (256)
            Decoder(256 + 256, 256, 128, 3, 1),   # x3 (256) + d3 (256)
            Decoder(128 + 128, 128, 64, 3, 1)     # x2 (128) + d2 (128)
        ]

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        x1 = self.encoders[0](x)
        x2 = self.encoders[1](self.maxpool(x1))
        x3 = self.encoders[2](self.maxpool(x2))
        x4 = self.encoders[3](self.maxpool(x3))
        x5 = self.encoders[4](self.maxpool(x4))
        d5 = self.decoders[0](self.maxpool(x5))
        d4 = self.decoders[1](torch.cat([d5, x5], dim=1))
        d3 = self.decoders[2](torch.cat([d4, x4], dim=1))
        d2 = self.decoders[3](torch.cat([d3, x3], dim=1))
        d1 = self.decoders[4](torch.cat([d2, x2], dim=1))
        return self.final(torch.cat([d1, x1], dim=1))

class UnetVgg11(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vgg = torchvision.models.vgg11(pretrained=False).features
        self.maxpool = torch.nn.MaxPool2d(2)
        self.encoders = [
            torch.nn.Sequential(self.vgg[0:2]),   # 64
            torch.nn.Sequential(self.vgg[3:5]),   # 128
            torch.nn.Sequential(self.vgg[6:10]),  # 256
            torch.nn.Sequential(self.vgg[11:15]), # 512
            torch.nn.Sequential(self.vgg[16:20])  # 512
        ]

        self.decoders = [
            Decoder(512, 512, 256, 3, 1),         # x5 (512)
            Decoder(512 + 256, 512, 256, 3, 1),   # x5 (512) + d5 (256)
            Decoder(512 + 256, 512, 256, 3, 1),   # x4 (512) + d4 (256)
            Decoder(256 + 256, 256, 128, 3, 1),   # x3 (256) + d3 (256)
            Decoder(128 + 128, 128, 64, 3, 1)     # x2 (128) + d2 (128)
        ]

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        import pdb; pdb.set_trace()
        x1 = self.encoders[0](x)
        x2 = self.encoders[1](self.maxpool(x1))
        x3 = self.encoders[2](self.maxpool(x2))
        x4 = self.encoders[3](self.maxpool(x3))
        x5 = self.encoders[4](self.maxpool(x4))
        d5 = self.decoders[0](self.maxpool(x5))
        d4 = self.decoders[1](torch.cat([d5, x5], dim=1))
        d3 = self.decoders[2](torch.cat([d4, x4], dim=1))
        d2 = self.decoders[3](torch.cat([d3, x3], dim=1))
        d1 = self.decoders[4](torch.cat([d2, x2], dim=1))
        return self.final(torch.cat([d1, x1], dim=1))

class UnetResnet34(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=False)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.encoders = [
            torch.nn.Sequential(
                self.resnet.conv1,
                self.resnet.bn1,
                self.resnet.relu
            ),                   # 64
            self.resnet.layer1,  # 64
            self.resnet.layer2,  # 128
            self.resnet.layer3,  # 256
            self.resnet.layer4   # 512
        ]

        self.decoders = [
            Decoder(512, 512, 256, 3, 1),         # x5 (512)
            Decoder(512 + 256, 512, 256, 3, 1),   # x5 (512) + d5 (256)
            Decoder(256 + 256, 512, 256, 3, 1),   # x4 (256) + d4 (256)
            Decoder(128 + 256, 256, 128, 3, 1),   # x3 (128) + d3 (256)
            Decoder(64 + 128, 128, 64, 3, 1),     # x2 (64) + d2 (128)
            Decoder(64 + 64, 128, 64, 3, 1)       # x1 (64) + d1 (64)
        ]

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        x1 = self.encoders[0](x)
        x2 = self.encoders[1](self.maxpool(x1))
        x3 = self.encoders[2](x2)
        x4 = self.encoders[3](x3)
        x5 = self.encoders[4](x4)
        d5 = self.decoders[0](self.maxpool(x5))
        d4 = self.decoders[1](torch.cat([d5, x5], dim=1))
        d3 = self.decoders[2](torch.cat([d4, x4], dim=1))
        d2 = self.decoders[3](torch.cat([d3, x3], dim=1))
        d1 = self.decoders[4](torch.cat([d2, x2], dim=1))
        d0 = self.decoders[5](torch.cat([d1, x1], dim=1))
        return self.final(d0)

class ResidualDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True)
        )

        self.upsampler = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.upsampler(self.layers(x) + x)

class ResidualUnet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=False)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.encoders = [
            torch.nn.Sequential(
                self.resnet.conv1,
                self.resnet.bn1,
                self.resnet.relu
            ),                   # 64
            self.resnet.layer1,  # 64
            self.resnet.layer2,  # 128
            self.resnet.layer3,  # 256
            self.resnet.layer4   # 512
        ]

        self.decoders = [
            ResidualDecoder(512, 256, 3, 1),         # x5 (512)
            ResidualDecoder(512 + 256, 256, 3, 1),   # x5 (512) + d5 (256)
            ResidualDecoder(256 + 256, 256, 3, 1),   # x4 (256) + d4 (256)
            ResidualDecoder(128 + 256, 128, 3, 1),   # x3 (128) + d3 (256)
            ResidualDecoder(64 + 128, 64, 3, 1),     # x2 (64) + d2 (128)
            ResidualDecoder(64 + 64, 64, 3, 1)       # x1 (64) + d1 (64)
        ]

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        x1 = self.encoders[0](x)
        x2 = self.encoders[1](self.maxpool(x1))
        x3 = self.encoders[2](x2)
        x4 = self.encoders[3](x3)
        x5 = self.encoders[4](x4)
        d5 = self.decoders[0](self.maxpool(x5))
        d4 = self.decoders[1](torch.cat([d5, x5], dim=1))
        d3 = self.decoders[2](torch.cat([d4, x4], dim=1))
        d2 = self.decoders[3](torch.cat([d3, x3], dim=1))
        d1 = self.decoders[4](torch.cat([d2, x2], dim=1))
        d0 = self.decoders[5](torch.cat([d1, x1], dim=1))
        return self.final(d0)


if __name__ == '__main__':
    input = torch.zeros((1, 3, 128, 128))
    model = ResidualUnet(2)
    model(input)
