import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F

import os
from PIL import Image
import pandas as pd


class FusionNet_importance(nn.Module):
    """
    Weighted approach
    """
    def __init__(self, model, meta_in=3, meta_out=3, dropout=0.3, hidden = 5):
        super(FusionNet_importance, self).__init__()
        self.img_extractor = nn.Sequential(*list(model.children())[:-1], nn.Dropout(p=dropout)) 
        num_features = list(model.modules())[-1].in_features
        self.img_extractor.requires_grad_(False) 
       
        self.metadata_extractor = nn.Sequential(
            #layer order: L -> BN -> ReLU: Zhang2020: https://arxiv.org/pdf/2004.08955.pdf; He2016: https://arxiv.org/pdf/1603.05027.pdf
            nn.Linear(meta_in, hidden),
            nn.Dropout(p=dropout), 
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden, meta_out),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(meta_out),
            nn.ReLU(inplace=True)
        )
    
        self.fc_img = nn.Linear(num_features, 2) # two outputs, probability and importance
        self.fc_meta = nn.Linear(meta_out, 2)
        self.fc = nn.Linear(num_features + meta_out, 2) # a value for each class 
        
    def forward(self, img, meta):
    
        img = self.img_extractor(img).flatten(1) # img features,z.B. 2048 
        meta = self.metadata_extractor(meta).flatten(1) # meta features z.B. 3 
        
        # probs/importance of each channel
        img = self.fc_img(img)
        meta = self.fc_meta(meta)
        
        probs_img = img[:,0].sigmoid().unsqueeze(1) 
        probs_meta = meta[:,0].sigmoid().unsqueeze(1)
        probs = torch.cat((probs_img, probs_meta), dim=1)
        
        imps_img = img[:,1].unsqueeze(1) # additional dimension necessary 
        imps_meta = meta[:,1].unsqueeze(1)
        importance = torch.cat((imps_img, imps_meta), dim=1)
        importance = importance.softmax(dim=1)
        
        # fusion
        x = (importance * probs).sum(dim=1) # hadamard product of importance and probs, summation for each sample in the batch
        
        return x 
        

class FusionNet_SEMul(nn.Module):
    """
    SE-approach
    """
    def __init__(self, model, meta_in=3, meta_out=3, dropout=0.3, hidden1 = 5, dropout1=0.5):
        super(FusionNet_SEMul, self).__init__()
        print(meta_in, hidden1, dropout, meta_out, dropout1)
        self.img_extractor = nn.Sequential(*list(model.children())[:-1], nn.Dropout(p=dropout)) # only the extractor part of the pretrained model 
        num_features = list(model.modules())[-1].in_features
        self.img_extractor.requires_grad_(False) # set the gradients false from the start
  
        self.metadata_extractor = nn.Sequential(
            nn.Linear(meta_in, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
          
            nn.Linear(hidden1, meta_out),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(meta_out),
            nn.ReLU(inplace=True) 
        )      

        self.fc = nn.Sequential(nn.Dropout(p=dropout1),nn.Linear(num_features, 1)) 
        
    def forward(self, img, meta): 
        img = self.img_extractor(img).flatten(1)
        meta = self.metadata_extractor(meta).flatten(1).sigmoid()  
        fusion = img*meta
        x = self.fc(fusion).squeeze(1).sigmoid() # remove unnecessary dimension 
        return x 


class FusionNet(nn.Module):
    """
    CAT-approach
    """
    def __init__(self, model, meta_in=3, meta_out=3, dropout=0.3, hidden1 = 5, hidden2 = 5, reduction = 512, dropout1=0.5):
        super(FusionNet, self).__init__()
        self.img_extractor = nn.Sequential(*list(model.children())[:-1])
        num_features = list(model.modules())[-1].in_features
        self.img_extractor.requires_grad_(False) 
        
        self.metadata_extractor = nn.Sequential(
            nn.Linear(meta_in, hidden1),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
          
            nn.Linear(hidden1, meta_out),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(meta_out),
            nn.ReLU(inplace=True)
        )
        # reduction is not used
        self.reduction = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, reduction)
        )
        
        self.fc = nn.Sequential(nn.Dropout(p=dropout1),nn.Linear(num_features + meta_out, 2)) 
        
    def forward(self, img, meta):
        img = self.img_extractor(img).flatten(1)
        meta = self.metadata_extractor(meta).flatten(1) 
        fusion = torch.cat([img, meta], dim=1)
        x = self.fc(fusion)
        return x

