#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""


# put the WB_sRGB where you run your code or add path to it
import sys
sys.path.append('./WB_sRGB/WB_sRGB_Python/') 

# also you can copy the models directory from WB_sRGB to where you run this code
# for easy access


import numpy as np
import matplotlib.pyplot as plt
from flip import computeFLIP
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from classes import WBsRGB as wb_srgb


def gray_world(im):
    img = np.copy(im)
    # implement this function
    return img

def white_world(im):
    imw = np.copy(im)
    # implement this function
    return imw



if __name__ == "__main__":
    # load data
    
    im_bad = plt.imread('abbey_badcolor.jpg').astype(np.float32)/255
    im_ok = plt.imread('abbey_correct.jpg').astype(np.float32)/255
    
    plt.imshow(np.hstack((im_bad,im_ok)))
    plt.show()
    
    
    im_gray = gray_world(im_bad)
    im_white = white_world(im_bad)
    
    
    
    upgraded_model = 1
    gamut_mapping = 2
    
    # call the whitebalance method of Afifi
   
    
    # im_afifi =  ...
    
    plt.imshow(np.hstack((im_gray,im_white,im_afifi)))
    plt.show()
    
    
    flip_gray,_ = ...
    flip_white,_ = ...
    flip_afifi,_ = ...
    ssim_gray = ...
    ssim_white = ...
    ssim_afifi = ...
    
    psnr_gray = ...
    psnr_white = ...
    psnr_afifi = ...
    
    
    print(flip_gray)
    print(flip_white)
    print(flip_afifi)
    
    print(ssim_gray)
    print(ssim_white)
    print(ssim_afifi)
    
    print(psnr_gray)
    print(psnr_white)
    print(psnr_afifi)
    
    
    

