import os
import numpy as np
import torch
from PIL import Image
import yaml
from torchvision import transforms, datasets

class Synthia(torch.utils.data.Dataset):
    def __init__(self, path_list, train = 'train', data_mode = 'rgb', scale = 1, target_width = 572, target_height = 572):
        
        self.mode = train
        self.data_mode = data_mode
        self.scale = scale
        self.target_w = target_width
        self.target_h = target_height
        
        # Get image file paths
        with open(path_list, 'r') as file:
            self.imgs = file.readlines()
            
        # Replace file paths for images with GT
        self.masks = [
            path.replace("/RGBLeft/", "/GTLeftDebug/")
            for path in self.imgs
        ]

        # Load in learning maps for each channel
        with open("synthia_3channels.yaml", 'r') as stream:
            synthiayaml = yaml.safe_load(stream)
        self.learning_map_red = synthiayaml['learning_map_red']
        self.learning_map_green = synthiayaml['learning_map_green']
        self.learning_map_blue = synthiayaml['learning_map_blue']
        
        self.names = synthiayaml['name']
        

    def get_names(self):
        return self.names
        
    
    # Change values in the red channel since labels are encoded in red channel
    def convert_label(self, label, inverse = False):
        tempr = label[0,:,:].copy()*255
        tempg = label[1,:,:].copy()*255
        tempb = label[2,:,:].copy()*255
        
        modified = np.full((572, 572), False)
        
        for k, v in self.learning_map_red.items():
            label[0][tempr==k] = v
            modified = np.logical_or(modified, tempr==k)
        for k, v in self.learning_map_green.items():
            label[0][np.logical_and(tempg==k, np.logical_not(modified))] = v
            modified = np.logical_or(modified, tempg==k)
        for k, v in self.learning_map_blue.items():
            label[0][np.logical_and(tempb==k, np.logical_not(modified))] = v 
            modified = np.logical_or(modified, tempb==k)
        return label
     
     
    def __getitem__(self, index):
        img_path = self.imgs[index].rstrip()
        mask_path = self.masks[index].rstrip()
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        w,h = img.size
        target_w, target_h = int(w * self.scale), int(h * self.scale)
        target_w, target_h = self.target_w, self.target_h
        img = img.resize((target_w, target_h))
        img = transforms.ToTensor()(img)
        
        mask = mask.resize((target_w, target_h))
        mask = transforms.ToTensor()(mask)
        
        mask = np.array(mask)
        # Removed because convert_label implementation needed all 3 channels (RGB)
        # mask = mask [0,:,:]
        
        mask = self.convert_label(mask)        
        mask = mask.astype(np.uint8)
        
        img = np.asarray(img)
        
        return torch.as_tensor(img.copy()).float().contiguous(), torch.as_tensor(mask.copy()).long().contiguous()
        
        
    def __len__(self):
        return len(self.imgs)