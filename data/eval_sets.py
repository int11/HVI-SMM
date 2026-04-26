import os
import torch.utils.data as data
import torch.nn.functional as F
from data.util import list_images, load_img


def pad_to_multiple(t, factor=8):
    """Reflect-pad a CHW tensor so H,W are multiples of `factor`.
    Returns (padded_tensor, original_h, original_w)."""
    h, w = t.shape[1], t.shape[2]
    H = ((h + factor) // factor) * factor
    W = ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    if padh == 0 and padw == 0:
        return t, h, w
    t = F.pad(t.unsqueeze(0), (0, padw, 0, padh), 'reflect').squeeze(0)
    return t, h, w


class SingleFolderEvalDataset(data.Dataset):
    """Eval dataset over a flat image folder. Pads to multiple of 8."""

    def __init__(self, data_dir, transform=None, pad_factor=8):
        super().__init__()
        self.data_filenames = list_images(data_dir)
        self.transform = transform
        self.pad_factor = pad_factor

    def __getitem__(self, index):
        img = load_img(self.data_filenames[index])
        file = os.path.basename(self.data_filenames[index])
        h, w = None, None
        if self.transform:
            img = self.transform(img)
            img, h, w = pad_to_multiple(img, factor=self.pad_factor)
        return img, file, h, w

    def __len__(self):
        return len(self.data_filenames)


class PairedEvalDataset(data.Dataset):
    """Eval dataset over `data_dir/{folder1, folder2}` paired by sorted name."""

    def __init__(self, data_dir, folder1='low', folder2='high', transform=None):
        super().__init__()
        self.low_paths = list_images(os.path.join(data_dir, folder1))
        self.high_paths = list_images(os.path.join(data_dir, folder2))
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.low_paths[index])
        gt = load_img(self.high_paths[index])
        file = os.path.basename(self.low_paths[index])
        if self.transform:
            input = self.transform(input)
            gt = self.transform(gt)
        return input, gt, file

    def __len__(self):
        return len(self.low_paths)
