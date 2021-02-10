import os
import torch
import torchvision
import random
import warnings

import numpy as np

from torch.utils.data import Dataset
from PIL import Image


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class CustomizedDataset_Fusion(Dataset):
    def __init__(self, path_set, metadata, rs_tiles, nr_tiles, transform = None, non_image = None, encoding_scheme = None):
       
        self.path_set = path_set
        self.csv_files = metadata
        self.rs_tiles = rs_tiles
        self.nr_tiles = nr_tiles
        self.transform = transform
        self.non_image = non_image
        
        self.samples = self.get_images()
        self.encoding_scheme = encoding_scheme 
    
        
    def __len__(self):
        return len(self.samples)
    
    def get_images(self): # get all tiles of the slides in path_set 
        samples = []
        for slide_path in self.path_set: 
            tiles_of_slide = []
            for root, dirs, files in os.walk(slide_path): 
                for file in files: 
                    if isinstance(file, str):
                        file = bytes(file, 'utf-8')
                    file = file.decode('utf-8')
                    path = os.path.join(root, file)
                    tiles_of_slide.append(path)
            
            # take only some tiles
            if self.nr_tiles != 0:
                random.seed(self.rs_tiles)
                s = random.sample(tiles_of_slide, min(len(tiles_of_slide), self.nr_tiles))
                samples.append(s)
            # take all tiles 
            else: 
                samples.append(tiles_of_slide)
        samples = np.asarray(samples)
        samples = np.concatenate(samples)
        return samples 
    
    def __getitem__(self, idx):
        """
        non_image parameters decides whether only images, or images and patient data are returned 
        """
        img_path = self.samples[idx]
        image = Image.open(img_path) # thats a tile
        slide_name = img_path.split('/')[-2] # Slide names look like these: 1) 2019-02-04 15.20.16_col_row_4_11.jpg or 2) F01__10_2.jpg
        row_idx = self.csv_files.slide_name[self.csv_files.slide_name == slide_name].index.tolist()[0]
        label = self.csv_files['label'][row_idx]
        
        # slide assignment of the tile 
        slide_id = int(row_idx) # slide id corresponds with row index in metadata-Dataframe 
        
        # position assignment of the tile within the slide 
        row = int(img_path.split('_')[-1].split('.')[0])
        col = int(img_path.split('_')[-2])
                
        label = (label, slide_id, col, row)
        label = torch.tensor(label)

        if self.transform: 
            image = self.transform(image)
        
        if self.non_image != None: 
            if self.encoding_scheme == 'scale01' or self.encoding_scheme == 'unscaled' or self.encoding_scheme == 'classified_2':
                non_image_data = torch.zeros(len(self.non_image)) 
                for counter, key in enumerate(self.non_image): 
                    non_image_data[counter] = self.csv_files[key][row_idx]
                
            if self.encoding_scheme == 'onehot':
                non_image_data = []
                for key in self.non_image: 
                    non_image_data.append(self.csv_files[key][row_idx])
                non_image_data = np.concatenate(non_image_data)
                non_image_data = torch.Tensor(non_image_data)
                #print(non_image_data)
                    
            sample = image, non_image_data, label
            
        else: 
            sample = image, label
        
        return sample 
        
        
        
        
        