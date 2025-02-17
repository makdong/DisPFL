import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
import pdb

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1): 
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) 
        out = F.relu(out) 
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, class_num=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # print(block)
        # print(num_blocks[0])
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, class_num)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        # print(strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  
            self.in_planes = planes * block.expansion  
        return nn.Sequential(*layers)  

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out) 
        out = self.layer2(out) 
        out = self.layer3(out) 
        out = self.layer4(out) 
        out = F.avg_pool2d(out, 4)  
        out = out.view(out.size(0), -1)  
        out = self.linear(out) 
        return out



def customized_resnet18(pretrained: bool = False, class_num=10,progress: bool = True) -> ResNet:
    res18 = ResNet(BasicBlock, [2, 2, 2, 2],class_num=class_num)


    # Change BN to GN
    res18.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

    res18.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

    res18.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

    # print(dict(res18.named_parameters()).keys())
    # print("\n\n\n\n\n")
    # print(res18.state_dict().keys())
    # print("\n\n\n\n\n")
    # print(len(dict(res18.named_parameters()).keys()))
    # print(len(res18.state_dict().keys()))
    assert len(dict(res18.named_parameters()).keys()) == len(res18.state_dict().keys()), 'More BN layers are there...'

    return res18


class tiny_ResNet(nn.Module):
    # 定义初始函数，输入参数为残差块，残差块数量，默认参数为分类数10
    def __init__(self, block, num_blocks, class_num=10):
        super(tiny_ResNet, self).__init__()
        # 设置第一层的输入通道数
        self.in_planes = 64

        # 定义输入图片先进行一次卷积与批归一化，使图像大小不变，通道数由3变为64得两个操作
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 定义第一层，输入通道数64，有num_blocks[0]个残差块，残差块中第一个卷积步长自定义为1
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # 定义第二层，输入通道数128，有num_blocks[1]个残差块，残差块中第一个卷积步长自定义为2
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # 定义第三层，输入通道数256，有num_blocks[2]个残差块，残差块中第一个卷积步长自定义为2
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # 定义第四层，输入通道数512，有num_blocks[3]个残差块，残差块中第一个卷积步长自定义为2
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 定义全连接层，输入512*block.expansion个神经元，输出10个分类神经元
        self.linear = nn.Linear(512 * block.expansion, class_num)

    # 定义创造层的函数，在同一层中通道数相同，输入参数为残差块，通道数，残差块数量，步长
    def _make_layer(self, block, planes, num_blocks, stride):
        # strides列表第一个元素stride表示第一个残差块第一个卷积步长，其余元素表示其他残差块第一个卷积步长为1
        strides = [stride] + [1] * (num_blocks - 1)
        # 创建一个空列表用于放置层
        layers = []
        # 遍历strides列表，对本层不同的残差块设置不同的stride
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # 创建残差块添加进本层
            self.in_planes = planes * block.expansion  # 更新本层下一个残差块的输入通道数或本层遍历结束后作为下一层的输入通道数
        return nn.Sequential(*layers)  # 返回层列表

    # 定义前向传播函数，输入图像为x，输出预测数据
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积和第一个批归一化后用ReLU函数激活
        out = self.layer1(out)  # 第一层传播
        out = self.layer2(out)  # 第二层传播
        out = self.layer3(out)  # 第三层传播
        out = self.layer4(out)  # 第四层传播
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)  # 全连接传播
        return out

def tiny_resnet18(pretrained: bool = False, class_num=10,progress: bool = True) -> ResNet:
    res18 = tiny_ResNet(BasicBlock, [2, 2, 2, 2],class_num=class_num)


    # Change BN to GN
    res18.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

    res18.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

    res18.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

    assert len(dict(res18.named_parameters()).keys()) == len(
        res18.state_dict().keys()), 'More BN layers are there...'

    return res18
