# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:43:13 2020

@author: Administrator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.utils.model_zoo as model_zoo
import numpy as np
import functions

'''
model_v33是在model_v32上的改进，删减一层
'''
        

class DSPNet(nn.Module):
    def __init__(self):
        super(DSPNet,self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
        self.fpn = FPN()
        self.up_input = PSPUpsample(4,4)
                
    def forward(self,x):
        x_4 = self.avgpool(x)
        x_16 = self.avgpool(x_4)            
        concat_4_sub_x = functions.concatenate_input_noise_map(x)
        x_4sub = self.up_input(concat_4_sub_x)
        concat_x_x_4sub = torch.cat([x,x_4sub],1)    
        x_16_out,x_4_out,x_1_out= self.fpn(concat_x_x_4sub)        
        x_16_out = x_16_out + x_16
        x_4_out = x_4_out + x_4       
        x_1_out = x_1_out + x
        
        return x_16_out,x_4_out,x_1_out
        
class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p) 

class ASPP(nn.Module):
    def __init__(self,channel):
        super(ASPP,self).__init__()
    
        self.conv_3 = nn.Sequential(
            nn.Conv2d(channel,channel//2,kernel_size=3,stride=1,padding=3,dilation=3),nn.BatchNorm2d(channel//2),nn.ReLU())

        self.conv_6 = nn.Sequential(
            nn.Conv2d(channel,channel//2,kernel_size=3,stride=1,padding=6,dilation=6),nn.BatchNorm2d(channel//2),nn.ReLU())
        
        self.avg_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(channel,channel//2,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(channel//2),nn.ReLU())
        
        self.conv_out = nn.Sequential(
                nn.Conv2d(3*(channel//2),channel,kernel_size=1))
        
        
        
    def forward(self,x):
        h,w = x.size(2),x.size(3) 
        
        out_3 = self.conv_3(x)
        out_6 = self.conv_6(x)
        out_avg = F.upsample(self.avg_pool(x),size=(h,w),mode='bilinear')

        out = torch.cat([out_3, out_6, out_avg], 1)
        out = self.conv_out(out)
        
        return out

class AS3_64(nn.Module):
    def __init__(self,channel):
        super(AS3_64,self).__init__()

        self.conv_1x1_1 = nn.Sequential(
                nn.Conv2d(channel,channel,kernel_size=3, stride=1, padding=1, dilation=1),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )        

        self.conv_3x3_2 = nn.Sequential(
                nn.Conv2d(channel,channel,kernel_size=3, stride=1, padding=3, dilation=3),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )

        self.conv_3x3_3 = nn.Sequential(
                nn.Conv2d(channel,channel,kernel_size=3, stride=1, padding=6, dilation=6),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )
        
        self.conv_out = nn.Sequential(
                nn.Conv2d(channel*3,channel,kernel_size=1),nn.BatchNorm2d(channel),nn.ReLU()
                )
        
        
    def forward(self,x):
        out_3x3_1 = self.conv_1x1_1(x)    
        out_3x3_2 = self.conv_3x3_2(x)
        out_3x3_3 = self.conv_3x3_3(x)

        out = torch.cat([out_3x3_1, out_3x3_2, out_3x3_3], 1)
        out = self.conv_out(out)
        
        return out

class AS_AM_64(nn.Module):
    def __init__(self, channel):
        super(AS_AM_64,self).__init__()
        
        self.conv_in = nn.Sequential(
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )
        
        self.conv_mid = nn.Sequential(
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU() ,  
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU() 
                )
        
        self.conv_sigmoid = nn.Sequential(
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.Sigmoid()
                )
        
        self.conv_out = nn.Sequential(
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )
        
        self.as3 = AS3_64(channel)
        
    def forward(self,x_in):
        x = self.conv_in(x_in)
        x = self.as3(x)
        x = self.conv_mid(x)
        
        x = self.conv_sigmoid(x) *x_in + x_in
        x_out = self.conv_out(x)
        
        return x_out


class AS3_128(nn.Module):
    def __init__(self,channel):
        super(AS3_128,self).__init__()

        self.conv_1x1_1 = nn.Sequential(
                nn.Conv2d(channel,channel,kernel_size=3, stride=1, padding=1, dilation=1),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )        

        self.conv_3x3_2 = nn.Sequential(
                nn.Conv2d(channel,channel,kernel_size=3, stride=1, padding=3, dilation=3),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )

        self.conv_3x3_3 = nn.Sequential(
                nn.Conv2d(channel,channel,kernel_size=3, stride=1, padding=6, dilation=6),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )
        
        self.conv_out = nn.Sequential(
                nn.Conv2d(channel*3,channel,kernel_size=1),nn.BatchNorm2d(channel),nn.ReLU()
                )

        
    def forward(self,x):
        out_1x1_1 = self.conv_1x1_1(x)  
        out_3x3_2 = self.conv_3x3_2(x)
        out_3x3_3 = self.conv_3x3_3(x)

        out = torch.cat([out_1x1_1, out_3x3_2, out_3x3_3], 1)
        out = self.conv_out(out)
        
        return out
class AS_AM_128(nn.Module):
    def __init__(self, channel):
        super(AS_AM_128,self).__init__()
        
        self.conv_in = nn.Sequential(
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )
        
        self.conv_mid = nn.Sequential(
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU()  ,
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU()  
                )
        
        self.conv_sigmoid = nn.Sequential(
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.Sigmoid()
                )
        
        self.conv_out = nn.Sequential(
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )
        
        self.as3 = AS3_128(channel)
        
    def forward(self,x_in):
        x = self.conv_in(x_in)
        x = self.as3(x)
        x = self.conv_mid(x)      
        x = self.conv_sigmoid(x) *x_in + x_in
        x_out = self.conv_out(x)
        
        return x_out
        
class AS3_256(nn.Module):
    def __init__(self,channel):
        super(AS3_256,self).__init__()

        self.conv_1x1_1 = nn.Sequential(
                nn.Conv2d(channel,channel,kernel_size=3, stride=1, padding=1, dilation=1),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )        

        self.conv_3x3_2 = nn.Sequential(
                nn.Conv2d(channel,channel,kernel_size=3, stride=1, padding=3, dilation=3),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )

        self.conv_3x3_3 = nn.Sequential(
                nn.Conv2d(channel,channel,kernel_size=3, stride=1, padding=6, dilation=6),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )
        
        self.conv_out = nn.Sequential(
                nn.Conv2d(channel*3,channel,kernel_size=1),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )
   
        
    def forward(self,x):
        out_3x3_1 = self.conv_1x1_1(x)      
        out_3x3_2 = self.conv_3x3_2(x)
        out_3x3_3 = self.conv_3x3_3(x)      
        
        out = torch.cat([out_3x3_1, out_3x3_2, out_3x3_3], 1)
        out = self.conv_out(out)
        
        return out

class AS_AM_256(nn.Module):
    def __init__(self,channel):
        super(AS_AM_256,self).__init__()
        
        self.conv_in = nn.Sequential(
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )
        
        self.conv_mid = nn.Sequential(
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU(),     
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )
        
        self.conv_sigmoid = nn.Sequential(
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.Sigmoid()
                )
        
        self.conv_out = nn.Sequential(
                nn.Conv2d(channel, channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU()
                )
        
        self.as3 = AS3_256(channel)
        
    def forward(self,x_in):
        x = self.conv_in(x_in)
        x = self.as3(x)
        x = self.conv_mid(x)      
        x = self.conv_sigmoid(x) *x_in + x_in
        x_out = self.conv_out(x)
        
        return x_out
 

class FEAM(nn.Module):
    def __init__(self, channel,First=False):
        super(FEAM,self).__init__()
        if First:
            self.conv_in = nn.Sequential(
                nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False),nn.LeakyReLU())
        else:
            self.conv_in = nn.Sequential(
                nn.Conv2d(channel*3,channel,kernel_size=3,stride=1,padding=1,bias=False),nn.LeakyReLU())
        self.conv = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU(),
            ASPP(channel),
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(channel),nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False),nn.Sigmoid())

    def forward(self,x_in):
        x = self.conv_in(x_in)
        x = self.conv(x) * x + x
        return x
    
class FPN(nn.Module):
    def __init__(self, channel_in = 5, channel_out = 1, channel_1 = 16, channel_2 = 32, channel_3 = 64):
        super(FPN, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(channel_in,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_1,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_1,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        
        self.conv2_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(channel_1,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_2,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_2,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        
        self.conv3_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(channel_2,channel_3,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_3,channel_3,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_3,channel_3,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        
        self.FEAM_64 = AS_AM_64(channel_1)       
        self.FEAM_128 = AS_AM_128(channel_2)
        self.FEAM_256 = AS_AM_256(channel_3)
        
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(channel_3*2,channel_3,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_3,channel_3,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_3,channel_3,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_3,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        
        self.conv3_out = nn.Sequential(
            nn.Conv2d(channel_2,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_2,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_2,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_1,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_1,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_1,channel_out,kernel_size=3,stride=1,padding=1))
        
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(channel_3,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_2,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_2,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_2,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())   
            
        self.conv4_out = nn.Sequential(
            nn.Conv2d(channel_1,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_1,channel_out,kernel_size=3,stride=1,padding=1))
        
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(channel_2,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_1,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_1,channel_out,kernel_size=3,stride=1,padding=1)) 
            
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(channel_3+channel_2,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(channel_2+channel_1,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())

    def _upsample(self,x,y):
        _,_,H,W = y.size()
        return F.upsample(x,size=(H,W),mode='bilinear')
    def forward(self,x):
        #encoder
        x1_1 = self.conv1_1(x)
        x2_1 = self.conv2_1(x1_1)
        x3_1 = self.conv3_1(x2_1)
        #middle
        x3_2 = self.FEAM_256(x3_1)
        x3_3 = torch.cat([x3_2,x3_1],1)
        
        x3_4 = self.conv3_2(x3_3)
        
        x3_5 = self._upsample(x3_4,x2_1)
        x3_0 = self.FEAM_128(self.conv4_2(torch.cat([x2_1,self._upsample(x3_2,x2_1)],1)))
        x4_1 = self.conv4_1(torch.cat([x3_5,x3_0],1))
        
        x4_2 = self._upsample(x4_1,x1_1)
        x4_0 = self.FEAM_64(self.conv5_2(torch.cat([x1_1,self._upsample(x3_0,x1_1)],1)))
        x_out = self.conv5_1(torch.cat([x4_2,x4_0],1))
        
        x_16_out = self.conv3_out(x3_4)
        
        x_4_out = self.conv4_out(x4_1)

        return x_16_out, x_4_out, x_out   