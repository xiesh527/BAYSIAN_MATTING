import cv2
import numpy as np

# Input the image and its trimap and get them to float32 format and divide them by 255 for the matting process below 
img = cv2.imread('img.jpg').astype(np.float32) / 255.0
trimap = cv2.imread('trimap.png').astype(np.float32) / 255.0

# Input the ground truth to calculate the MSE
groundtruth = cv2.imread('groundtruth.png').astype(np.float32) / 255.0
groundtruth = groundtruth[:, :, 0]

# Determine the foreground, background, and unknown pixels in trimap by the fixed thresholds which are obtained from observing the trimap matrix.
foreg = trimap > 50
backg = trimap < 1
# Determine unknown pixels in the grey area in trimap
unkwn = ~(foreg | backg)

# Get the width, height and the number of channel of the image, these parameters will be used in the calculation of the alpha matte.
# Channel numbers should be 3 because the image is the RGB colour image.
width, height, channel = img.shape

# Create the Laplacian matrix by using the cv2.Laplacian() function in OpenCV
# cv2.Laplacian() is an OpenCV function which returns a discrete approximation of Laplace’s differential operator applied to the input image
lpmt = cv2.Laplacian(img, cv2.CV_32F)

# Calculation process of alpha matte
alpha = np.zeros((width, height), dtype=np.float32) # Initialize a alpha as the same size of image
for i in range(channel): # Get the alpha matte for each channel
    alpha[unkwn[:, :, 0]] += lpmt[unkwn[:, :, 0], i] ** 2
    # Get the square of alpha to make sure there are no negative numbers

# Divide the alpha by 3(channel number) to get the mean value of alpha, get the square root of it, then 1 minus it to obtain the correct alpha matte
alpha = 1 - np.sqrt(alpha / channel)
alpha[backg[:, :, 0]] = 0 # Set alpha of the background pixels as 0
alpha[foreg[:, :, 0]] = 1 # Set alpha of the foreground pixels as 1

# Use this alpha matte to process the input image then get the final laplacian matting result, then save the output alpha matte and matting result as a .png format image.
outimg = alpha[..., np.newaxis] * img
cv2.imwrite('output_alpha.png', (alpha * 255).astype(np.uint8))
cv2.imwrite('output_image.png', (outimg * 255).astype(np.uint8))

# Calculate the MSE between the alpha matte and the ground truth
MSE = np.mean((alpha - groundtruth) ** 2)

# Print the MSE 
print(MSE)
