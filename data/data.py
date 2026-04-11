from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Resize, Lambda
from PIL import Image
from data.LOLdataset import LOLv1DatasetFromFolder, LOLv2DatasetFromFolder, LOLv2SynDatasetFromFolder
from data.eval_sets import *
from data.SICE_blur_SID import *

def transform1(size=256):
    resizer = Resize(size)
    return Compose([
        Lambda(lambda img: resizer(img) if min(img.size) < size else img),
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])

def transform2():
    def resize_to_multiple_of_8(img):
        w, h = img.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), resample=Image.BICUBIC)
        return img
    return Compose([Lambda(resize_to_multiple_of_8), ToTensor()])

def transform3():
    return Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])
