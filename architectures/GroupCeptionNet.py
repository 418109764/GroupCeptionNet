import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )
        self.output_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def grouped_conv3x3(in_planes, out_planes, stride=1, groups=None):
    if groups is None: groups = in_planes // 32
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = grouped_conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = grouped_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class GroupCeptionNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(GroupCeptionNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define (2+Inception) layer structure
        self.layer1 = self._make_layer(block, 64, layers[0], use_inception=True,
                                       inception_channels=[16, 16, 32, 4, 8, 8])  # 输出 64 通道
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_inception=True,
                                       inception_channels=[32, 32, 64, 8, 16, 16])  # 输出 128 通道
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_inception=True,
                                       inception_channels=[64, 64, 128, 16, 32, 32])  # 输出 256 通道
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_inception=True,
                                       inception_channels=[128, 128, 256, 32, 64, 64])  # 输出 512 通道

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, use_inception=False, inception_channels=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # Add the first BasicBlock for changing size
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        # Use the Inception module instead of the second BasicBlock to adjust the number of output channels
        if use_inception:
            layers.append(Inception(self.inplanes, *inception_channels))
        else:
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.classifier(x)
        x = self.fc1(x)
        return x


def groupCeptionNet(pretrained=False, **kwargs):
    return GroupCeptionNet(BasicBlock, [2, 2, 2, 2], num_classes=2)

# import torch.onnx
# model = airnext()
# x= torch.randn(1, 3, 224, 224)
# model = model.to('cuda')
# summary(model, (3, 224, 224))
# torch.onnx.export(model, x, "airnext.onnx", input_names=["input"], output_names=["output"])
# import netron
# netron.start("airnext.onnx")
