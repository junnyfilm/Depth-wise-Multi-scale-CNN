import torch
import torch.nn as nn

def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool,self).__init__()
    def forward(self,x):
        return x.mean(axis=-1) 
    
class Feature_extractor(nn.Module):

    def __init__(self):
        super(Feature_extractor, self).__init__()
        self.bn = nn.BatchNorm1d(1)
        self.conv1 = nn.Sequential(nn.Conv1d(1, 3, kernel_size=1, stride = 1, padding=0),
                                 nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(1, 3, kernel_size=3, stride = 1, padding=1),
                                 nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(1, 3, kernel_size=5, stride = 1, padding=2),
                                 nn.ReLU())
        
    def forward(self, x):
        x = self.bn(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        return x1,x2,x3
    
class backbone(nn.Module):
    def __init__(self):
        super(backbone,self).__init__()
        self.bn_concat = nn.BatchNorm1d(30)

        self.bottle_1 = nn.Conv1d(30,128,kernel_size=1, stride=1, bias=False)
        self.bottle_2 = nn.Conv1d(128,64, kernel_size=3, stride=1, bias=False, padding=1)

        self.bottle_3 = nn.Conv1d(30,128,kernel_size=3, stride=1, bias=False, padding=1)
        self.bottle_4 = nn.Conv1d(128, 64, kernel_size=1, stride=1, bias=False)

        self.bn_bottleneck24 = nn.BatchNorm1d(30+64+64)

        self.bottle_5 = nn.Conv1d(30+64+64,256,kernel_size=1, stride=1, bias=False)
        self.bottle_6 = nn.Conv1d(256,128,kernel_size=3, stride=1, bias=False, padding=1)

        self.bn_output = nn.BatchNorm1d(128)
        self.conv_out = nn.Conv1d(128, 64, kernel_size=1, stride=1, bias=False)
        self.relu =  nn.ReLU()
        self.gap = GlobalAvgPool()
        

        self.fc1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU())
        self.fc2 =nn.Linear(32, 11)

    def forward(self, inputs):
        concat = self.relu(inputs)
        bn_concat = self.bn_concat(concat)

        bottle_1 = self.bottle_1(bn_concat)
        bottle_1 = self.relu(bottle_1)
        bottle_2 = self.bottle_2(bottle_1)
        bottle_2 = self.relu(bottle_2)

        bottle_3 = self.bottle_3(bn_concat)
        bottle_3 = self.relu(bottle_3)
        bottle_4 = self.bottle_4(bottle_3)
        bottle_4 = self.relu(bottle_4)

        bottle24_concat = torch.cat([bn_concat, bottle_2, bottle_4],axis=1)
        bottle24_concat = self.bn_bottleneck24(bottle24_concat)

        bottle_5 = self.bottle_5(bottle24_concat)
        bottle_5 = self.relu(bottle_5)
        bottle_6 = self.bottle_6(bottle_5)
        bottle_6 = self.relu(bottle_6)

        conv_out = self.bn_output(bottle_6)
        out = self.conv_out(conv_out)
        out = self.relu(out)
        outview = self.gap(out)

        out = self.fc1(outview)
        out = self.fc2(out)

        return out,outview


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
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
    
class ResNet(nn.Module):
    def __init__(self, block, layers, in_channel=30, out_channel=11, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv1d(in_channel, 32, kernel_size=2, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, 3, stride=1)
        self.layer2 = self._make_layer(block, 128, 3, stride=1)
        self.layer3 = self._make_layer(block, 256, 3, stride=1)
        # self.layer4 = self._make_layer(block, 512, 3, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.gap = GlobalAvgPool()
        
        # self.fc1 = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.fc3 =nn.Linear(128, 11)
        self.fc3.weight.data.normal_(0, 0.005)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x=torch.squeeze(x,3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.gap(x)
        x_view = x.view(x.size(0),-1)
        # x = self.fc1(x_view)
        x = self.fc2(x_view)
        x = self.fc3(x)

        return x,x_view
    
class SE_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d((1))
        self.excitation = nn.Sequential(
            nn.Linear(30, 1),
            nn.ReLU(),
            nn.Linear(1, 30),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1) 
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1),1)
        return x