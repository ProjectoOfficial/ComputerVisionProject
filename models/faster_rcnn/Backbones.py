import torch
from torch import nn
from ResidualBlock import ResidualBlock
from torchvision.ops import MultiScaleRoIAlign

class Resnet(nn.Module):

    def __init__(self, in_channels:int = 3, out_channels: int = 10) -> None:
        super(Resnet, self).__init__()

        self.rb1 = ResidualBlock(in_channels, 64)
        self.rb2 = ResidualBlock(64, 128,)
        self.rb3 = ResidualBlock(128, 256,)
        self.rb4 = ResidualBlock(256, 512)
        self.rb5 = ResidualBlock(512, 512)

        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        return x
