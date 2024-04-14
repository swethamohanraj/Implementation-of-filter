# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import cv2, matplotlib.py libraries and read the saved images using cv2.imread().

### Step2
Convert the saved BGR image to RGB using cvtColor().

### Step3
By using the following filters for image smoothing:filter2D(src, ddepth, kernel), Box filter,Weighted Average filter,GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]), medianBlur(src, ksize),and for image sharpening:Laplacian Kernel,Laplacian Operator.

### Step4
Apply the filters using cv2.filter2D() for each respective filters.

### Step5
Plot the images of the original one and the filtered one using plt.figure() and cv2.imshow().

## Program:
```python
Developed By : K.M.Swetha
Register Number: 212221240055

import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("kore.png")
original_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

## 1. Smoothing Filters
# i) Using Averaging Filter
kernel1 = np.ones((11,11),np.float32)/121
avg_filter = cv2.filter2D(original_image,-1,kernel1)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(avg_filter)
plt.title("Filtered")
plt.axis("off")

# ii) Using Weighted Averaging Filter
kernel2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
weighted_filter = cv2.filter2D(original_image,-1,kernel2)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(weighted_filter)
plt.title("Filtered")
plt.axis("off")

# iii) Using Gaussian Filter
gaussian_blur = cv2.GaussianBlur(src = original_image, ksize = (11,11), sigmaX=0, sigmaY=0)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Filtered")
plt.axis("off")

# iv) Using Median Filter
median = cv2.medianBlur(src=original_image,ksize = 11)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(median)
plt.title("Filtered")
plt.axis("off")

## 2. Sharpening Filters
# i) Using Laplacian Kernel
kernel3 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
laplacian_kernel = cv2.filter2D(original_image,-1,kernel3)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian_kernel)
plt.title("Filtered")
plt.axis("off")

# ii) Using Laplacian Operator
laplacian_operator = cv2.Laplacian(original_image,cv2.CV_64F)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian_operator)
plt.title("Filtered")
plt.axis("off")
```
## OUTPUT:
### 1. Smoothing Filters

## i) Using Averaging Filter
![image](https://github.com/swethamohanraj/Implementation-of-filter/assets/94228215/b89e35b9-297b-4673-93c0-3a5beba7e0e4)


## ii) Using Weighted Averaging Filter
![image](https://github.com/swethamohanraj/Implementation-of-filter/assets/94228215/8cf8a439-3759-45b8-b5af-5b845cc237bd)


## iii) Using Gaussian Filter
![image](https://github.com/swethamohanraj/Implementation-of-filter/assets/94228215/bc73e315-b6e0-49f4-947b-1d33e1db9be6)

## iv) Using Median Filter
![image](https://github.com/swethamohanraj/Implementation-of-filter/assets/94228215/ce3aa1ce-3a68-4c90-ab90-9db1395a90e5)



### 2. Sharpening Filters

## i) Using Laplacian Kernal
![image](https://github.com/swethamohanraj/Implementation-of-filter/assets/94228215/f80083c2-2642-46c6-b3b3-da6828dcdb72)

## ii) Using Laplacian Operator
![image](https://github.com/swethamohanraj/Implementation-of-filter/assets/94228215/0df61c4a-4be9-4836-ac8f-c339b878df75)


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
