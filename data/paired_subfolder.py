import os
import random
import torch.utils.data as data
from data.util import list_images, load_img
from data.base import PairedTransformMixin


def _scan_subdirs(parent):
    if not os.path.isdir(parent):
        return []
    subs = [os.path.join(parent, d) for d in os.listdir(parent)
            if os.path.isdir(os.path.join(parent, d))]
    subs.sort()
    return subs


class PairedSubfolderDataset(PairedTransformMixin, data.Dataset):
    """Input/GT both partitioned into subfolders.

    Patterns:
      - ``gt_mode='per_file'``: pick a random subfolder, pick the same index
        in both input/GT subfolders (LOLBlur).
      - ``gt_mode='single'``: pick a random subfolder, pick a random input
        file but always use the first file in the GT subfolder (SID).

    `length` controls the epoch length (defaults to number of non-empty
    input subfolders).  Empty subfolders are filtered up-front so we no
    longer need a runtime ``while True`` loop.
    """

    def __init__(self, data_dir, input_subdir, gt_subdir, gt_mode='per_file',
                 transform=None, length=None):
        super().__init__()
        if gt_mode not in ('per_file', 'single'):
            raise ValueError(f"gt_mode must be 'per_file' or 'single', got {gt_mode!r}")
        self.data_dir = data_dir
        self.transform = transform
        self.gt_mode = gt_mode

        input_root = os.path.join(data_dir, input_subdir)
        gt_root = os.path.join(data_dir, gt_subdir)

        entries = []
        for sub in _scan_subdirs(input_root):
            name = os.path.basename(sub)
            gt_sub = os.path.join(gt_root, name)
            if not os.path.isdir(gt_sub):
                continue
            in_files = list_images(sub)
            gt_files = list_images(gt_sub)
            if not in_files or not gt_files:
                continue
            entries.append((in_files, gt_files))
        if not entries:
            raise ValueError(
                f'No usable subfolders under {input_root} / {gt_root}')
        self.entries = entries
        self.length = length if length is not None else len(entries)

    def __getitem__(self, index):
        in_files, gt_files = self.entries[random.randint(0, len(self.entries) - 1)]
        i = random.randint(0, len(in_files) - 1)
        in_path = in_files[i]
        if self.gt_mode == 'per_file':
            gt_path = gt_files[i] if i < len(gt_files) else gt_files[-1]
        else:  # 'single'
            gt_path = gt_files[0]
        im1 = load_img(in_path)
        im2 = load_img(gt_path)
        file1 = os.path.basename(in_path)
        file2 = os.path.basename(gt_path)
        im1, im2 = self._apply_paired_transform(im1, im2)
        return im1, im2, file1, file2

    def __len__(self):
        return self.length


class SubfolderWithFlatGTDataset(PairedTransformMixin, data.Dataset):
    """Input lives in subfolders under ``data_dir``, GT is a flat folder
    ``{parent_of_data_dir}/{label_dir_name}/{subfolder_name}.JPG`` (SICE).
    """

    def __init__(self, data_dir, label_dir_name='label', label_ext='.JPG',
                 transform=None, length=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        parent, _ = os.path.split(data_dir.rstrip(os.sep))
        label_root = os.path.join(parent, label_dir_name)

        entries = []
        for sub in _scan_subdirs(data_dir):
            name = os.path.basename(sub)
            in_files = list_images(sub)
            if not in_files:
                continue
            gt_path = os.path.join(label_root, name + label_ext)
            if not os.path.isfile(gt_path):
                # try common alternate extensions
                alt = None
                for ext in ('.JPG', '.jpg', '.png', '.jpeg', '.bmp'):
                    cand = os.path.join(label_root, name + ext)
                    if os.path.isfile(cand):
                        alt = cand
                        break
                if alt is None:
                    continue
                gt_path = alt
            entries.append((in_files, gt_path))
        if not entries:
            raise ValueError(
                f'No usable subfolders under {data_dir} matched against {label_root}')
        self.entries = entries
        self.length = length if length is not None else len(entries)

    def __getitem__(self, index):
        in_files, gt_path = self.entries[random.randint(0, len(self.entries) - 1)]
        i = random.randint(0, len(in_files) - 1)
        in_path = in_files[i]
        im1 = load_img(in_path)
        im2 = load_img(gt_path)
        file1 = os.path.basename(in_path)
        file2 = os.path.basename(gt_path)
        im1, im2 = self._apply_paired_transform(im1, im2)
        return im1, im2, file1, file2

    def __len__(self):
        return self.length
