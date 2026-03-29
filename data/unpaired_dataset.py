import os
import random
import glob
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import is_image_file, load_img


class UnpairedDataset(data.Dataset):
    """
    FoFA 비지도 학습용 Unpaired 데이터셋.
    low 이미지와 high 이미지를 독립적으로 샘플링하여
    런타임에 랜덤으로 매칭한다.

    반환 형식: (im_low, im_high, file_low, file_high)
    - im_low: ExDark 등 저조도 이미지
    - im_high: COCO 등 정상조도 이미지
    현재의 paired 로더와 동일한 배치 인터페이스를 유지한다.
    """
    def __init__(self, low_dir, high_dir, transform=None):
        super(UnpairedDataset, self).__init__()
        self.transform = transform

        # Support recursive search for datasets with subfolders like ExDark
        def get_all_images(target_dir):
            all_files = []
            for root, _, files in os.walk(target_dir):
                for f in files:
                    if is_image_file(f):
                        all_files.append(join(root, f))
            return sorted(all_files)

        self.low_files = get_all_images(low_dir)
        self.high_files = get_all_images(high_dir)

        if len(self.low_files) == 0:
            raise ValueError(f'No images found in low_dir: {low_dir}')
        if len(self.high_files) == 0:
            raise ValueError(f'No images found in high_dir: {high_dir}')

    def __getitem__(self, index):
        # low 이미지는 index 순서대로, high 이미지는 랜덤 샘플링
        low_path = self.low_files[index % len(self.low_files)]
        high_path = self.high_files[random.randint(0, len(self.high_files) - 1)]

        im_low = load_img(low_path)
        im_high = load_img(high_path)

        _, file_low = os.path.split(low_path)
        _, file_high = os.path.split(high_path)

        if self.transform:
            seed = np.random.randint(1000000)
            random.seed(seed)
            torch.manual_seed(seed)
            im_low = self.transform(im_low)
            # high는 독립적으로 변환 (다른 seed)
            seed2 = np.random.randint(1000000)
            random.seed(seed2)
            torch.manual_seed(seed2)
            im_high = self.transform(im_high)

        return im_low, im_high, file_low, file_high

    def __len__(self):
        return len(self.low_files)
