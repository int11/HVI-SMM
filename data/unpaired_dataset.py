import os
import random
import torch.utils.data as data
from os.path import join
from data.util import is_image_file, load_img
from data.base import PairedTransformMixin


class UnpairedDataset(PairedTransformMixin, data.Dataset):
    """
    FoFA 비지도 학습용 Unpaired 데이터셋.
    low 이미지와 high 이미지를 독립적으로 샘플링하여
    런타임에 랜덤으로 매칭한다.

    반환 형식: (im_low, im_high, file_low, file_high)
    - im_low: ExDark 등 저조도 이미지
    - im_high: COCO 등 정상조도 이미지

    transform 은 두 이미지에 **서로 다른 seed** 로 적용된다 (paired
    데이터셋의 동기화 transform 과는 의도적으로 다르다).
    """

    def __init__(self, low_dir, high_dir, transform=None):
        super().__init__()
        self.transform = transform

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
        low_path = self.low_files[index % len(self.low_files)]
        high_path = self.high_files[random.randint(0, len(self.high_files) - 1)]

        im_low = load_img(low_path)
        im_high = load_img(high_path)

        file_low = os.path.basename(low_path)
        file_high = os.path.basename(high_path)

        # 의도적으로 두 이미지에 서로 다른 seed 적용 (unpaired)
        im_low = self._apply_independent_transform(im_low)
        im_high = self._apply_independent_transform(im_high)

        return im_low, im_high, file_low, file_high

    def __len__(self):
        return len(self.low_files)
