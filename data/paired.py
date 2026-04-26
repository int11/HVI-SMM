import os
import torch.utils.data as data
from data.util import list_images, load_img
from data.base import PairedTransformMixin


class PairedFlatFolderDataset(PairedTransformMixin, data.Dataset):
    """Paired dataset where `data_dir/low_folder` and `data_dir/high_folder`
    contain matched images (sorted 1:1 by filename). Replaces
    `LOLDatasetFromFolder`."""

    def __init__(self, data_dir, low_folder, high_folder, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.folder1 = os.path.join(data_dir, low_folder)
        self.folder2 = os.path.join(data_dir, high_folder)
        self.data_filenames = list_images(self.folder1)
        self.data_filenames2 = list_images(self.folder2)

    def __getitem__(self, index):
        im1 = load_img(self.data_filenames[index])
        im2 = load_img(self.data_filenames2[index])
        file1 = os.path.basename(self.data_filenames[index])
        file2 = os.path.basename(self.data_filenames2[index])
        im1, im2 = self._apply_paired_transform(im1, im2)
        return im1, im2, file1, file2

    def __len__(self):
        return len(self.data_filenames)
