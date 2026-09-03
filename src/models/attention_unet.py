# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .attention_gate import AttentionGate

class AttUnet(nn.Module):
    def __init__(self, out_channels, x_channels, q_channels, attention_type):
        super(AttUnet, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()          
        self.sigmoid = nn.Sigmoid()     
                    
        self.conv2d_1 = nn.Conv2d(160, 96, kernel_size=3, stride=1, padding=1)                  
        self.conv2d_2 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1)       
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)        
                                                                     
        self.conv2d_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2d_5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv2d_6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2d_7 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv2d_8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2d_9 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv2d_10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2d_11 = nn.Conv2d(160, 64, kernel_size=3, stride=1, padding=1)
        self.conv2d_12 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.ag1 = AttentionGate(x_channels[0], q_channels[0], attention_type[0])
        self.ag2 = AttentionGate(x_channels[1], q_channels[1], attention_type[1])
        self.ag3 = AttentionGate(x_channels[2], q_channels[2], attention_type[2])
        
        self.avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  
        

    def forward(self, x0):                                 # in 32, 160,128,128  
        x1 = self.conv2d_1(x0); x1 = self.leakyrelu(x1);      #    32, 96,128,128  
        x2 = self.avgpool2d(x1)                           #    32,96, 64, 64  
        
        x2 = self.conv2d_2(x2); x2 = self.leakyrelu(x2);      #    32,64,   64, 64  
        x3 = self.avgpool2d(x2)                           #    32,64,   32, 32 
        
        x3 = self.conv2d_3(x3); x3 = self.leakyrelu(x3);       #    32, 64 32, 32 
        x4 = self.avgpool2d(x3)                           #    32, 64, 16, 16

        x4 = self.conv2d_4(x4); x4 = self.leakyrelu(x4)                    #    32, 64, 16, 16
        x4 = self.conv2d_5(x4); x4 = self.leakyrelu(x4)                    #    32, 64, 16, 16
        
        xag1 = self.ag1(x3,x4)                                         #    32, 64, 32, 32
        
        x5 = self.upsample(x4)                                           #    32, 64, 32, 32
        x5 = self.conv2d_6(x5); x5 = self.leakyrelu(x5)           #    32, 64, 32, 32
        x5 = self.conv2d_7(torch.cat((x5, xag1), dim=1)); x5 = self.leakyrelu(x5)                    #    32, 64, 32, 32
        
        xag2 = self.ag2(x2,x4)                             #    32, 64, 64, 64
        
        x5 = self.upsample(x5)                                           #    32, 64, 64, 64
        x5 = self.conv2d_8(x5); x5 = self.leakyrelu(x5)                    #    32, 64, 64, 64
        x5 = self.conv2d_9(torch.cat((x5, xag2), dim=1)); x5 = self.leakyrelu(x5)                    #    32, 64, 64, 64
        
        xag3 = self.ag3(x1,x4)                            #    32, 96, 64, 64
        
        x5 = self.upsample(x5)                                           #    32, 64, 128, 128
        x5 = self.conv2d_10(x5); x5 = self.leakyrelu(x5)            #    32, 64, 128, 128
        x5 = self.conv2d_11(torch.cat((x5, xag3), dim=1)); x5 = self.leakyrelu(x5)            #    32, 64, 128, 128
        x5 = self.conv2d_12(x5);             #    32, 20, 128, 128
        return x5
    
    
    
