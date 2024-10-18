#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""


# put the WB_sRGB where you run your code or add path to it
# import sys
# sys.path.append('WB_sRGB/WB_sRGB_Python/classes/') 

# also you can copy the models directory from WB_sRGB to where you run this code
# for easy access

import cv2
import numpy as np
import matplotlib.pyplot as plt
from flip import computeFLIP
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from classes import WBsRGB as wb_srgb


def gray_world(im):
    img = np.copy(im)

    # Cumpute average value of each color channel across all pixels
    avg_rgb = np.mean(img, axis=(0, 1))
    # Divide global average intensity across each channel for an adjustment factor
    scale = avg_rgb.mean() / avg_rgb
    
    # Scale!
    img = img * scale
    # Error check, clip into range.
    img = np.clip(img, 0, 1)
    return img

def white_world(im):
    img = np.copy(im)

    # Get max
    max_rgb = np.max(img, axis=(0, 1))
    # Get scaling factor by inverting the maximum values
    scale = 1 / max_rgb
    
    # Scale!
    img = img * scale
    # Error check, clip into range.
    img = np.clip(img, 0, 1) 
    return img


if __name__ == "__main__":
    # load data
    
    im_bad = plt.imread('abbey_badcolor.jpg').astype(np.float32)/255
    im_ok = plt.imread('abbey_correct.jpg').astype(np.float32)/255
    print(im_bad.shape)
    plt.imshow(np.hstack((im_bad,im_ok)))
    plt.show()
    
    
    im_gray = gray_world(im_bad)
    im_white = white_world(im_bad)

    # create an instance of the WB model
    gamut_mapping = 2
    upgraded_model = 1
    wbModel = wb_srgb.WBsRGB(gamut_mapping=gamut_mapping,
                            upgraded=upgraded_model)
    I = plt.imread('abbey_badcolor.jpg').astype(np.float32)/255
    Ibgr = I[:,:,[2,1,0]]
    # White balance image
    im_afifi = wbModel.correctImage(I)  
    
    plt.imshow(np.hstack((im_gray,im_white,im_afifi[:,:,[2,1,0]])))
    plt.show()
    
    
    flip_gray,_ = computeFLIP(im_ok, im_gray)
    flip_white,_ = computeFLIP(im_ok, im_white)
    flip_afifi,_ = computeFLIP(im_ok, im_afifi)
    print("Done afifi")
    ssim_gray = ssim(im_ok, im_gray, channel_axis=-1, data_range=1.0)
    ssim_white = ssim(im_ok, im_white, channel_axis=-1, data_range=1.0)
    ssim_afifi = ssim(im_ok, im_afifi, channel_axis=-1, data_range=1.0)
    print("Done ssim")
    psnr_gray = psnr(im_ok, im_gray)
    psnr_white = psnr(im_ok, im_white)
    psnr_afifi = psnr(im_ok, im_afifi)
    print("Done psnr")
    
    
    print(flip_gray)
    print(flip_white)
    print(flip_afifi)
    
    print(ssim_gray)
    print(ssim_white)
    print(ssim_afifi)
    
    print(psnr_gray)
    print(psnr_white)
    print(psnr_afifi)
    
    
    

