import cv2
import numpy as np
from General_lib import compositing

def lap(img, trimap):
# Input the image and trimap
  img = img.astype(np.float32) / 255.0
  trimap = trimap.astype(np.float32) / 255.0

# Input the ground truth to calculate the MSE
  #groundtruth = cv2.imread('groundtruth.png').astype(np.float32) / 255.0
  #groundtruth = groundtruth[:, :, 0]

# Determine the foreground, background, and unknown pixels in trimap by the fixed thresholds which are obtained from observing the trimap matrix.
  foreg = trimap > 0.9
  backg = trimap < 0.45
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
      alpha[unkwn] += lpmt[unkwn, i] ** 2
    # Get the square of alpha to make sure there are no negative numbers

# Divide the alpha by 3(channel number) to get the mean value of alpha, get the square root of it, then 1 minus it to obtain the correct alpha matte
  alpha = 1 - np.sqrt(alpha / channel)
  alpha[backg] = 0 # Set alpha of the background pixels as 0
  alpha[foreg] = 1 # Set alpha of the foreground pixels as 1
  #alpha[(alpha >= 0.9) & (alpha <= 0.95)] = 1
  alpha_1 = alpha * 255
  alpha_3 = np.zeros((width, height, channel), dtype=np.float32)
  alpha_3[ :, :, 0] = alpha_1
  alpha_3[ :, :, 1] = alpha_1
  alpha_3[ :, :, 2] = alpha_1

# Use this alpha matte to process the input image then get the final laplacian matting result, then save the output alpha matte and matting result as a .png format image.
  outimg = alpha_3 * img 
  outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)
  cv2.imwrite('lap_alpha.png', (alpha_1).astype(np.uint8))
  cv2.imwrite('lap_image.png', (outimg).astype(np.uint8))

# Calculate the MSE between the alpha matte and the ground truth
  #MSE = np.mean((alpha - groundtruth) ** 2)

# Print the MSE 
  #print(MSE)
  return alpha_1, outimg