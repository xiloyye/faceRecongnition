class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        #输入及输出都为3通道，不改变原始图片通道数
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
    def forward(self, x):
        a = F.relu(self.conv(x))
        a = F.softmax(a.view(a.size(0), -1), dim=1).view_as(a)
        x = x * a
        return x
    
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5))
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class sia_net(nn.Module):
    def __init__(self , model):
        super(sia_net, self).__init__()
        #取掉model的后两层
        self.fc1 = nn.Sequential(
                nn.Sequential(*list(model.children())[:-2]),
                nn.AdaptiveAvgPool2d(1))

        self.fc1_0 = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.Linear(1024, 512))
        self.attention1 = Attention(1)

    def forward_once(self, x):
        x = self.attention1(x)
        x = self.fc1(x)
        x = x.view(x.size()[0], -1) 
        feature = self.fc1_0(x)     #feature
        return feature
    
    def forward(self, input_l, input_r):
        feature_l = self.forward_once(input_l)
        feature_r = self.forward_once(input_r)
        return feature_l, feature_r

def load_resnet50():
    resnet = torchvision.models.resnet50()
    model = sia_net(resnet)
    return model
