from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Resize, Lambda
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
    return Compose([ToTensor()])

def transform3():
    return Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])
