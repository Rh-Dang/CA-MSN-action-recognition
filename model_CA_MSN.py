# Copyright (c) 2022 Tongji University. All rights reserved.
# Licensed under the MIT License.
from torch import nn
import torch
import math


class CA_MSN(nn.Module):
    def __init__(self, num_classes, dataset, seg, args, bias = True):
        super(CA_MSN, self).__init__()

        self.dim1 = 256  # hidden dim = 256
        self.dataset = dataset
        self.seg = seg  # frame num
        num_joint = 25   # node num
        bs = args.batch_size 
        if args.train:
            self.spa = self.one_hot(bs, num_joint, self.seg)   # one-hot encoding skeleton joints
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()      
            self.tem = self.one_hot(bs, self.seg, num_joint)    # one-hot encoding video frames
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()       
        else:
            self.spa = self.one_hot(32 * 5, num_joint, self.seg)  
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(32 * 5, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, 64*4, norm=False, bias=bias)  # embedding frame index  
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)   # embedding joint type
        self.joint_embed = embed(3, 64, norm=True, bias=bias)
        self.dif_embed = embed(3, 64, norm=True, bias=bias)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))         
        self.cnn = mslocal(self.dim1, self.dim1 * 2, bias=bias)     #  MS-TCN
        
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)  # compute adjacency matrix
        
        #CA-GCN
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias, chennal_attention=1)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias, chennal_attention=1)
        self.gcn3 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias, chennal_attention=1)
        self.gcn4 = gcn_spa(self.dim1, self.dim1, bias=bias, chennal_attention=1)   
        self.gcn5 = gcn_spa(self.dim1, self.dim1, bias=bias, chennal_attention=1) 
        self.fc = nn.Linear(self.dim1 * 2, num_classes)     

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels   
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)    


    def forward(self, input):
        
        # Dynamic Representation
        bs, step, dim = input.size() 
        num_joints = dim //3   
        input = input.view((bs, step, num_joints, 3))  
        input = input.permute(0, 3, 2, 1).contiguous()  # bs * 3 * num_joints * num_frames
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)  #velocity
        pos = self.joint_embed(input)         # embedded input
        tem1 = self.tem_embed(self.tem)       # embedded frame index
        spa1 = self.spa_embed(self.spa)       # embedded joint type
        dif = self.dif_embed(dif)             # embedded velocity input
        dy = pos + dif                        
        # Joint-level Module
        input= torch.cat([dy, spa1], 1)       # finnal input with velocity and frame index
        g = self.compute_g1(input)            # compute adjacency matix 
        input = self.gcn1(input, g)           # CA-GCN
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)
        input = self.gcn4(input, g)
        input = self.gcn5(input, g)
        # Frame-level Module
        input = input + tem1                 # + embedded frame index
        input = self.cnn(input)              # max pooling + 2 linear
        # Classification
        output = self.maxpool(input)       
        output = torch.flatten(output, 1)  
        output = self.fc(output)

        return output

    def one_hot(self, bs, spa, tem):  

        y = torch.arange(spa).unsqueeze(-1)   
        y_onehot = torch.FloatTensor(spa, spa) 

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)    

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)   
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)   

        return y_onehot

class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim* 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):    
    def __init__(self, dim = 3, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class local(nn.Module):  
    def __init__(self, dim1 = 3, dim2 = 3, bias = True , dila=1, pad=1):
        super(local, self).__init__()
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, pad), bias=bias,dilation=dila)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()    
        self.dropout = nn.Dropout2d(0.3)


    def forward(self, x1):
        x = self.cnn1(x1)+x1
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    

class mslocal(nn.Module):
    def __init__(self , dim1=2 , dim2=3 , bias = False):
        super(mslocal,self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20)) 
        #self.res=nn.Conv2d(dim1, dim2, kernel_size=1,bias=bias)
        self.branch1 = local(dim1, dim2, dila=1, pad=1)
        self.branch2 = local(dim1, dim2, dila=2, pad=2)
        self.branch3 = local(dim1, dim2, dila=3, pad=3)
        #self.branch4 = nn.Conv2d(dim1, dim1, kernel_size=1, bias=bias)
#         self.branch4_cnn=nn.Conv2d(dim1, dim1, kernel_size=1, bias=bias)
#         self.branch4_bn=nn.BatchNorm2d(dim1)
#         self.branch4_pool=nn.MaxPool2d(kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.full_conv =  nn.Conv2d(3*dim1, dim2, kernel_size=1,bias=bias)   
        self.bn = nn.BatchNorm2d(dim2)                  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)
        
    def forward(self, x):
        y=self.maxpool(x)
        #y_res= self.res(y)
        y1=self.branch1(y)
        y2=self.branch2(y)
        y3=self.branch3(y)
#         y4=self.branch4_cnn(y)
#         y4=self.branch4_bn(y4)
#         y4=self.relu(y4)
#         y4=self.branch4_pool(y4)
#         y4=self.branch4_bn(y4)
        y = torch.cat([y1,y2,y3],1)
        y=self.full_conv(y)
        #y=y+y_res
        y=self.bn(y)
        y=self.relu(y)
        y = self.dropout(y)
        return y

class gcn_spa(nn.Module):  
    def __init__(self, in_feature, out_feature, bias = False, reduction=1, chennal_attention=0):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)
        self.chennal_attention=chennal_attention
        ################################################
        if self.chennal_attention==1:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Sequential(    
                nn.Linear(out_feature, out_feature // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(out_feature // reduction, out_feature),
                nn.Sigmoid())
        ##################################################

    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()  #bs*3*num_joints*num_frames -> bs*num_frames*num_joints*3
        x = g.matmul(x)                         # (num_joints, num_joints) * (num_joints, 3)
        x = x.permute(0, 3, 2, 1).contiguous()  
        x = self.w(x)
        ##########SE-NET attention######
        if self.chennal_attention==1:
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc1(y).view(b, c, 1, 1)
            x=x*y
        #####################################
        x = x + self.w1(x1)
        x = self.relu(self.bn(x))
        return x

class compute_g_spa(nn.Module):   
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g
    
