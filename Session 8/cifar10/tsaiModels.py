import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Residual Block as in Assignment 8
    """
    def __init__(self, inchannels, outchannels, stride_=1):
        super().__init__()
        self.res = nn.Sequential(nn.Conv2d(in_channels=inchannels, out_channels=outchannels, 
                                         kernel_size=(3, 3),stride=stride_, padding=1, 
                                         bias=False),
                             nn.BatchNorm2d(outchannels),
                             nn.ReLU(),
                             nn.Conv2d(in_channels=inchannels, out_channels=outchannels, 
                                         kernel_size=(3, 3),stride=stride_, padding=1, 
                                         bias=False),
                             nn.BatchNorm2d(outchannels),
                             nn.ReLU()
                    )
    
    def forward(self,out):
        out = self.res(out)
        return out

class tsaiNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super().__init__()
        ## Preparation Block
        self.prep = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=64, 
                                  kernel_size=(3, 3),stride=1, padding=1, 
                                  bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                        )
        
        ## Layer 1
        self.l1x = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, 
                                             kernel_size=(3, 3),stride=1, padding=1, 
                                             bias=False),
                                 nn.MaxPool2d(2, 2),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU()
                        )
        self.resblock1 = ResBlock(inchannels = 128, outchannels = 128, stride_= 1)

        
        ## Layer 2
        self.l2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, 
                                             kernel_size=(3, 3),stride=1, padding=1, 
                                             bias=False),
                                 nn.MaxPool2d(2, 2),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU()
                        )
        
        ## Layer 3
        self.l3x = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, 
                                             kernel_size=(3, 3),stride=1, padding=1, 
                                             bias=False),
                                 nn.MaxPool2d(2, 2),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU()
                        )
        
        self.resblock2 = ResBlock(inchannels = 512, outchannels = 512, stride_= 1)
        
        self.max_4 = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        ## Preparation Layer
        x = self.prep(x)
        ## Layer 1
        x = self.l1x(x)
        res1 = self.resblock1(x)
        x = x + res1
        ## Layer 2
        x = self.l2(x)
        ## Layer 3
        x = self.l3x(x)
        res2 = self.resblock2(x)
        x = x+res2

        x = self.max_4(x)  
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


        
        
def tsaiResNet():
    """
    Custom ResNet model.
    """
    return tsaiNet(ResBlock)    