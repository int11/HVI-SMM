import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *
from torchvision import transforms as t

    
class LOLDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, folder1, folder2, transform=None):
        super(LOLDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        # if use_norm:
        #     self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.folder1 = join(data_dir, folder1)
        self.folder2 = join(data_dir, folder2)
        self.data_filenames = sorted([join(self.folder1, x) for x in listdir(self.folder1) if is_image_file(x)])
        self.data_filenames2 = sorted([join(self.folder2, x) for x in listdir(self.folder2) if is_image_file(x)])

    def __getitem__(self, index):
        im1 = load_img(self.data_filenames[index])
        im2 = load_img(self.data_filenames2[index])
        _, file1 = os.path.split(self.data_filenames[index])
        _, file2 = os.path.split(self.data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)  # make a seed with numpy generator 
        if self.transform:
            random.seed(seed)  # apply this seed to img tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2) 
        return im1, im2, file1, file2

    def __len__(self):
        return len(self.data_filenames)


# Backward compatibility classes
class LOLv1DatasetFromFolder(LOLDatasetFromFolder):
    def __init__(self, data_dir, transform=None):
        super().__init__(data_dir, 'low', 'high', transform)


class LOLv2DatasetFromFolder(LOLDatasetFromFolder):
    def __init__(self, data_dir, transform=None):
        super().__init__(data_dir, 'Low', 'Normal', transform)


class LOLv2SynDatasetFromFolder(LOLDatasetFromFolder):
    def __init__(self, data_dir, transform=None):
        super().__init__(data_dir, 'Low', 'Normal', transform)





