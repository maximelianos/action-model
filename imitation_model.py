import torch
import torch.nn as nn


"""
Imitation learning network
"""

def imginfo(array):
    print(type(array))
    print(array.dtype, array.shape)


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        TODO:
        Implement a standard ResNet-style convolutional block.

        Args:
            in_channels (int): Number of channels in the input feature map.
            out_channels (int): Number of channels produced by the block (also number of channels after 1st Conv2D layer).
            is_res (bool): Whether to include a residual connection.

        - Use two Conv2D layers with:
            - kernel size = 3
            - stride = 1
            - padding = 1
        - Each followed by BatchNorm and GELU activation.
        - Track if in_channels == out_channels (used for skip connection logic).
        '''
        self.same_channels = (in_channels == out_channels)
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # Hint: you may want to store:
        # self.same_channels
        # self.is_res
        # self.conv1 = nn.Sequential(...)
        # self.conv2 = nn.Sequential(...)

        #raise NotImplementedError("Define conv1, conv2, same_channels, and is_res here")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class PointConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        TODO:
        Implement a standard ResNet-style convolutional block.

        Args:
            in_channels (int): Number of channels in the input feature map.
            out_channels (int): Number of channels produced by the block (also number of channels after 1st Conv2D layer).
            is_res (bool): Whether to include a residual connection.

        - Use two Conv2D layers with:
            - kernel size = 1
            - stride = 1
            - padding = 1
        - Each followed by BatchNorm and GELU activation.
        - Track if in_channels == out_channels (used for skip connection logic).
        '''
        self.same_channels = (in_channels == out_channels)
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # Hint: you may want to store:
        # self.same_channels
        # self.is_res
        # self.conv1 = nn.Sequential(...)
        # self.conv2 = nn.Sequential(...)

        #raise NotImplementedError("Define conv1, conv2, same_channels, and is_res here")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class CNN(nn.Module):

    def __init__(self, input_channels=None, n_classes=3):
        super(CNN, self).__init__()
        self.emb_size = 64 * 4 * 4 # output channels * output h * output w 

        layers = [
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1), # (3, 48, 48) -> (16, 24, 24)
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (16, 24, 24) -> (32, 12, 12)
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (32, 12, 12) -> (64, 12, 12)
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.AdaptiveMaxPool2d((4, 4)),  # (16, 4, 4)
            nn.Flatten(),
            
            nn.Linear(self.emb_size, 128),
            nn.GELU(),
            nn.Linear(128, n_classes)
        ]
        self.model = nn.Sequential(
            *layers
        )

    def forward(self, x):
        # compute forward pass
        x = self.model(x)
        return x