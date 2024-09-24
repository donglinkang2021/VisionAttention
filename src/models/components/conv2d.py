import torch.nn as nn

def conv3x3(
        in_planes: int, 
        out_planes: int, 
        stride: int = 1, 
    ) -> nn.Conv2d:
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(
        in_planes: int, 
        out_planes: int, 
        stride: int = 1
    ) -> nn.Conv2d:
    """
    1x1 convolution
    """
    return nn.Conv2d(
        in_planes, 
        out_planes, 
        kernel_size=1, 
        stride=stride, 
        bias=False
    )

if __name__ == '__main__':
    import torch
    # Test conv1x1
    conv = conv1x1(in_planes=3, out_planes=64)
    print(conv)
    x = torch.randn(1, 3, 56, 56)
    y = conv(x)
    print(y.size())
    
    # Test conv3x3
    conv = conv3x3(in_planes=3, out_planes=64)
    print(conv)
    x = torch.randn(1, 3, 56, 56)
    y = conv(x)
    print(y.size())

# python -m src.models.components.conv2d