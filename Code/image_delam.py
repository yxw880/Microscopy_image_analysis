# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:16:42 2018

@author: wangy
"""

import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank


dir = "C:/Users/wangy/Documents/Python Scripts/"
file = 'sa31007_2-es03-ms03-mn01-micx8.jpg'

list_of_files = glob.glob('./*.jpg')   
path = os.path.join(dir,file)

img = cv2.imread(path,0)
plt.imshow(img, cmap = plt.cm.gray)

plt.hist(img.ravel(),256,[0,256]); plt.show()

ret,thresh1 = cv2.threshold(img,100,115,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,100,115,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,100,115,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,100,115,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,100,115,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

img_rescale = exposure.equalize_hist(img)

# Equalization
selem = disk(30)
img_eq = rank.equalize(img, selem=selem)

img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
img_adapteq = img_adapteq*255
#detect delamination

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img,(1,1),1000)
flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

bilateral_filtered_image = cv2.bilateralFilter(img, 5, 175, 175)
edge_detected_image = cv2.Canny(bilateral_filtered_image, 20,30)

##method2
##bilateral filter
blur = cv2.bilateralFilter(img,9,75,75)
plt.imshow(blur, cmap = plt.cm.gray)

##adaptive histogram eqaulization
equ = cv2.equalizeHist(blur)
plt.imshow(equ, cmap = plt.cm.gray)

##scikit module equalization
img_adapteq = exposure.equalize_adapthist(blur, clip_limit=0.1)
img_adapteq = img_adapteq*255
plt.imshow(img_adapteq, cmap = plt.cm.gray)

img_mthr = img_adapteq > 0.5
plt.imshow(img_mthr,cmap = plt.cm.gray)

kernel = np.ones((5,5),np.uint8)
img_mthr = np.array(img_mthr, dtype=np.uint8)
img_mthd = cv2.dilate(img_mthr,kernel,iterations = 1)
cv2.dilate(img_mthr,kernel,iterations = 1)

h, w = img_mthd.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
