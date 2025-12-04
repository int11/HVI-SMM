import os
import torch.utils.data as data
from os import listdir
from os.path import join
from data.util import *
import torch.nn.functional as F

class SICEDatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(SICEDatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.transform:
            input = self.transform(input)
            factor = 8
            h, w = input.shape[1], input.shape[2]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input = F.pad(input.unsqueeze(0), (0,padw,0,padh), 'reflect').squeeze(0)
        return input, file, h, w

    def __len__(self):
        return len(self.data_filenames)
    
    
class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, folder1='low', folder2='high', transform=None):
        super(DatasetFromFolderEval, self).__init__()
        # folder1/folder2 폴더 읽기
        low_dir = os.path.join(data_dir, folder1)
        high_dir = os.path.join(data_dir, folder2)
        low_files = [f for f in listdir(low_dir) if is_image_file(f)]
        high_files = [f for f in listdir(high_dir) if is_image_file(f)]
        low_files.sort()
        high_files.sort()
        self.low_paths = [os.path.join(low_dir, f) for f in low_files]
        self.high_paths = [os.path.join(high_dir, f) for f in high_files]  # high 폴더의 파일명 사용
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.low_paths[index])
        gt = load_img(self.high_paths[index])
        _, file = os.path.split(self.low_paths[index])
        if self.transform:
            input = self.transform(input)
            gt = self.transform(gt)
        return input, gt, file

    def __len__(self):
        return len(self.low_paths)