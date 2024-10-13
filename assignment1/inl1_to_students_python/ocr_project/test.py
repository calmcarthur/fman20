# import matplotlib.pyplot as plt
# import numpy as np

# data = np.random.random((10, 10))
# plt.imshow(data)
# plt.show()

    im = plt.imread('datasets/short1/im1.jpg')
    
    # read ground truth numbers
    gt_file = open('datasets/short1/im1.txt','r') 
    gt = gt_file.read()
    gt = gt[:-1] # remove newline character
    gt_file.close()
    
    # show image with ground truth
    plt.imshow(im)
    plt.title(gt)