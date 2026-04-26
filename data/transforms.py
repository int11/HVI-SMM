from PIL import Image
from torchvision.transforms import (
    Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip,
    Resize, Lambda,
)


def train_crop_transform(size=256):
    """Random crop + flip, with up-resize if image is too small (transform1)."""
    resizer = Resize(size)
    return Compose([
        Lambda(lambda img: resizer(img) if min(img.size) < size else img),
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])


def eval_pad8_transform():
    """Resize down to nearest multiple of 8 then ToTensor (transform2)."""
    def resize_to_multiple_of_8(img):
        w, h = img.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), resample=Image.BICUBIC)
        return img
    return Compose([Lambda(resize_to_multiple_of_8), ToTensor()])


def flip_only_transform():
    """Random flip only, no crop/resize (transform3)."""
    return Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])
