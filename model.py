import torch
import torch.nn as nn

def make_block(m, n, in_channels,out_channels,kernel_size,stride,padding=1):
    layer = []
    for i in range(m):
        if i == 1:
            layer.append('conv_'+n+'_'+i, nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding) 
            layer.append('relu_'+n+'_'+i, nn.ReLU(inplace=True))
        else: 
            layer.append('conv_'+n+'_'+i, nn.Conv2d(out_channels,out_channels,kernel_size,stride,padding)
            layer.append('relu_'+n+'_'+i, nn.ReLU(inplace=True))
    layer.append('pool'+n+'_stage1', nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    
    return nn.Sequential(layer)

class HandPoseModel(nn.Module):
    def __init__(self):
        
        super(HandPoseModel,self).__init__() 
        
        self.block1 = make_block(m=2,n=1,in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.block2 = make_block(m=2,n=2,in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.block3 = make_block(m=4,n=3,in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.block4 = make_block(m=4,n=4,in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.block5 = make_block(m=2,n=5,in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        
        self.conv5_3_CPM = nn.Conv2d(in_channels=512,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.relu5_3_CPM = nn.ReLU(inplace=True)
        
        self.conv_6_1 = nn.Conv2d(in_channels=128,out_channels=512,kernel_size=1,stride=1,padding=0)
        self.relu_6_1 = nn.ReLU(inplace=True)
        
        self.conv_6_2_CPM = nn.Conv2d(in_channels=512,out_channels=22,kernel_size=1,stride=1,padding=0)
        
        self.prev_stage2 = self.stage_block(2)
        self.prev_stage3 = self.stage_block(3)
        self.prev_stage4 = self.stage_block(4)
        self.prev_stage5 = self.stage_block(5)
        self.prev_stage6 = self.stage_block(6)


    def stage_block(self,stage,padding=1):
        layer = []
        for n in range(1,6):
            layer.append('Mconv{n}_stage{stage}'.format(n=n,stage=stage), nn.Conv2d(in_channels=150,out_channels=128,kernel_size=7,stride=1,padding=1))
            layer.append('Mrelu{n}_stage{stage}.format(n=n,stage=stage), nn.ReLU(inplace=True))
        layer.append('Mconv6_stage{}'.format(stage), nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,stride=1,padding=0))
        layer.append('Mrelu6_stage{}'.format(stage), nn.ReLU(inplace=True))
        layer.append('Mconv7_stage{}'.format(stage), nn.Conv2d(in_channels=128,out_channels=22,kernel_size=1,stride=1,padding=0))

        return nn.Sequential(layer)

    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        conv5_3_CPM = self.conv5_3_CPM(out)
        out = self.relu5_3_CPM(conv5_3_CPM)

        out = self.conv_6_1(out)
        out = self.relu_6_1(out)
        prev = self.conv_6_2_CPM(out)

        out = torch.cat([conv5_3_CPM,prev],1)

        prev = self.prev_stage2(out)
        out = torch.cat([conv5_3_CPM,prev],1)

        prev = self.prev_stage3(out)
        out = torch.cat([conv5_3_CPM,prev],1)

        prev = self.prev_stage4(out)
        out = torch.cat([conv5_3_CPM,prev],1)

        prev = self.prev_stage5(out)
        out = torch.cat([conv5_3_CPM,prev],1)

        out = self.prev_stage6(out)

        return out





        
        
        














        

