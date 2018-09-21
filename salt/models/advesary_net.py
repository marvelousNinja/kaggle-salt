import torch

class AdvesaryNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.AvgPool2d(32)
        )

        self.fc = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = self.layers(x)
        return self.fc(x.view(x.shape[0], -1))

if __name__ == '__main__':
    import pdb; pdb.set_trace()
