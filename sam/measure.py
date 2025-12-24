import os
import torch
import cv2
import warnings
# Filter out the torchvision pretrained parameter deprecation warnings
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum")
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import platform
import glob


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8))
    return psnr

# LPIPS 모델을 전역적으로 한 번만 생성
_lpips_model = None

def get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex', verbose=False)
        _lpips_model.cuda()
    return _lpips_model

def metrics(im_list, label_list, use_GT_mean):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    n = 0
    for item, label_item in list(zip(im_list, label_list)):
        n += 1
        
        # Handle different input types (PIL, numpy array, or file path)
        if isinstance(item, str):
            im1 = Image.open(item).convert('RGB')
        elif isinstance(item, Image.Image):
            im1 = item.convert('RGB')
        else:  # numpy array
            im1 = Image.fromarray((item * 255).astype(np.uint8))
            
        if isinstance(label_item, str):
            im2 = Image.open(label_item).convert('RGB')
        elif isinstance(label_item, Image.Image):
            im2 = label_item.convert('RGB')
        else:  # numpy array
            im2 = Image.fromarray((label_item * 255).astype(np.uint8))
            
        (h, w) = im2.size
        im1 = im1.resize((h, w))  
        im1 = np.array(im1) 
        im2 = np.array(im2)
        
        # Use metrics_one function
        score_psnr, score_ssim, score_lpips = metrics_one(im1, im2, use_GT_mean)
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips.item()
        torch.cuda.empty_cache()
    

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_lpips = avg_lpips / n
    return avg_psnr, avg_ssim, avg_lpips

def metrics_one(im1, im2, use_GT_mean):
    if isinstance(im1, Image.Image):
        im1 = np.array(im1)
    if isinstance(im2, Image.Image):
        im2 = np.array(im2)

    if use_GT_mean:
        mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
        mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
        im1 = np.clip(im1 * (mean_target/mean_restored), 0, 255)
    
    score_psnr = calculate_psnr(im1, im2)
    score_ssim = calculate_ssim(im1, im2)
    ex_p0 = lpips.im2tensor(im1).cuda()
    ex_ref = lpips.im2tensor(im2).cuda()
    
    # LPIPS 모델을 metrics_one에서 직접 가져옴
    lpips_model = get_lpips_model()
    score_lpips = lpips_model.forward(ex_ref, ex_p0)
    return score_psnr, score_ssim, score_lpips