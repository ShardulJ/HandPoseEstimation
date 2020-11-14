import torch
import torch.nn as nn
from collections import OrderedDict

def make_block(m, n, in_channels,out_channels,kernel_size,stride,padding=1):
    layer = []
    for i in range(1,m+1):
        if i == 1:
            layer.append(('conv'+str(n)+'_'+str(i), nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding))) 
            layer.append(('relu'+str(n)+'_'+str(i), nn.ReLU(inplace=True)))
        else: 
            layer.append(('conv'+str(n)+'_'+str(i), nn.Conv2d(out_channels,out_channels,kernel_size,stride,padding)))
            layer.append(('relu'+str(n)+'_'+str(i), nn.ReLU(inplace=True)))
    if n !=4:
        layer.append(('pool'+str(n)+'_stage1', nn.MaxPool2d(kernel_size=2,stride=2,padding=0)))
    
    return nn.Sequential(OrderedDict(layer))

class HandPoseModel(nn.Module):
    def __init__(self):
        
        super(HandPoseModel,self).__init__() 
        
        self.block1 = make_block(m=2,n=1,in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.block2 = make_block(m=2,n=2,in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.block3 = make_block(m=4,n=3,in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.block4 = make_block(m=4,n=4,in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.block5 = make_block(m=2,n=5,in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        

        self.conv5_3_CPM = nn.Conv2d(in_channels=512,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.relu5_4_stage1_3 = nn.ReLU(inplace=True)
        
        self.conv6_1_CPM = nn.Conv2d(in_channels=128,out_channels=512,kernel_size=1,stride=1,padding=0)
        self.relu6_4_stage1_1 = nn.ReLU(inplace=True)
        
        self.conv6_2_CPM = nn.Conv2d(in_channels=512,out_channels=22,kernel_size=1,stride=1,padding=0)
        
        self.prev_stage2 = self.stage_block(2)
        self.prev_stage3 = self.stage_block(3)
        self.prev_stage4 = self.stage_block(4)
        self.prev_stage5 = self.stage_block(5)
        self.prev_stage6 = self.stage_block(6)


    def stage_block(self,stage,padding=1):
        layer = []
        
        layer.append(('Mconv1_stage{stage}'.format(stage=stage), nn.Conv2d(in_channels=150,out_channels=128,kernel_size=7,stride=1,padding=3)))
        layer.append(('Mrelu1_2stage{stage}_1'.format(stage=stage), nn.ReLU(inplace=True)))
        for n in range(2,6):
            layer.append(('Mconv{n}_stage{stage}'.format(n=n,stage=stage), nn.Conv2d(in_channels=128,out_channels=128,kernel_size=7,stride=1,padding=3)))
            layer.append(('Mrelu1_{j}_stage{stage}_{n}'.format(j=n+1,stage=stage,n=n), nn.ReLU(inplace=True)))
        layer.append(('Mconv6_stage{}'.format(stage), nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,stride=1,padding=0)))
        layer.append(('Mrelu1_7_stage{}_6'.format(stage), nn.ReLU(inplace=True)))
        layer.append(('Mconv7_stage{}'.format(stage), nn.Conv2d(in_channels=128,out_channels=22,kernel_size=1,stride=1,padding=0)))

        return nn.Sequential(OrderedDict(layer))

    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        conv5_3_CPM = self.conv5_3_CPM(out)
        relu5_4_stage1_3 = self.relu5_4_stage1_3(conv5_3_CPM)
        
        out = self.conv6_1_CPM(relu5_4_stage1_3)
        out = self.relu6_4_stage1_1(out)
        
        prev = self.conv6_2_CPM(out)
        
        out1 = torch.cat([relu5_4_stage1_3,prev],1)
        prev = self.prev_stage2(out1)
        
        out2 = torch.cat([relu5_4_stage1_3,prev],1)
        prev = self.prev_stage3(out2)
        
        out3 = torch.cat([relu5_4_stage1_3,prev],1)
        prev = self.prev_stage4(out3)
        
        out4 = torch.cat([relu5_4_stage1_3,prev],1)
        prev = self.prev_stage5(out4)
        
        out5 = torch.cat([relu5_4_stage1_3,prev],1)
        out = self.prev_stage6(out5)

        return out

