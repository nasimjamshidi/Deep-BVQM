#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 20:21:52 2021

@author: nasimjamshidi
"""

import torch.nn as nn
import torch



class HeavyCNN(nn.Module):
            
            def __init__(self):
                super(HeavyCNN, self).__init__()
        
                self.pool = nn.MaxPool2d(2)
        
                self.conv1 = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                
                self.conv2 = nn.Sequential(
                    nn.Conv2d(16, 32, kernel_size=5, padding=2),
                    nn.ReLU(),
                )
                self.dropout = nn.Dropout(0.20)
        
                self.conv3 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.conv4 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=5, padding=2),
                    nn.ReLU(),
                )
               
                 
                self.fc1 = nn.Linear(11*11*64, 1000)
                self.fc2 = nn.Linear(1003, 5)
        
            def forward(self, x, additional1,additional2,additional3):
                out = self.pool(x)
                out = self.conv1(out)
                out = self.conv2(out)
                out = self.dropout(out)
                out = self.conv3(out)
                out = self.conv4(out)
                out = out.view(out.size(0), -1)
                out = self.fc1(out)
                newout = torch.cat((out, additional1,additional2,additional3), 1)
                out = self.fc2(newout)
        
                return out
        
class SimpleCNN(nn.Module):
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(22 * 22 * 32, 1000)
        self.fc2 = nn.Linear(1003, 5)

    def forward(self, x, additional1,additional2,additional3):
        #print(x.shape)
        out = self.conv1(x)
        #print(out.shape)
        out = self.conv2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        #print(out.shape)
        #add=additional1,additional2,additional3 
        newout = torch.cat((out, additional1,additional2,additional3 ), 1)
        out = self.fc2(newout)
        #print(out.shape)
        return out