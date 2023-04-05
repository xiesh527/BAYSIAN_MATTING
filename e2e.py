# Importing Dependencies
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import os
import cProfile

# Some important helper functions
from General_lib import Bayesian_Matte, compositing, disp_img
from laplacian import lap
import time
from skimage.metrics import structural_similarity as ssim

# Calculating the loss


def mse_loss(alpha, image_alpha):
    return np.sum(np.abs(alpha/255 - image_alpha/255)**2)/(alpha.shape[0]*alpha.shape[1])


# Reading the image
name = "GT12"
image_alpha = np.array(ImageOps.grayscale(Image.open(
    os.path.join("data", "gt_training_lowres", "{}.png".format(name)))))
save_name_origin = "ORI.png"
save_name_sharpen = "Sharpen.png"

# Reading the image and trimap
os.path.join("data", "gt_training_lowres", "{}.png".format(name))
image = np.array(Image.open(os.path.join(
    "data", "input_training_lowres", "{}.png".format(name))))
gaussian_image = cv2.GaussianBlur(image, (25, 25), 10.0)
unsharp_image = cv2.addWeighted(image, 1.5, gaussian_image, -0.5, 0)

# Timing the Bayesian Matting algorithm
Bay_start = time.time()
image_trimap = np.array(ImageOps.grayscale(Image.open(
    os.path.join("data", "trimap_training_lowres", "{}.png".format(name)))))
gaussian_trimap = cv2.GaussianBlur(image_trimap, (25, 25), 10)
unsharp_trimap = cv2.addWeighted(image_trimap, 1.5, gaussian_trimap, -0.5, 0)

# Running the Bayesian Matting algorithm
alpha, pixel_count = Bayesian_Matte(image, image_trimap)
unsharped_alpha, unsharped_pixel_count = Bayesian_Matte(
    unsharp_image, unsharp_trimap)
alpha *= 255
unsharped_alpha *= 255

# Converting to int for saving
alpha_int8 = np.array(alpha, dtype=int)
Bay_end = time.time()

# Saving the alpha channel
plt.imsave("{}_alpha.png".format(name), alpha, cmap='gray')
cv2.imwrite(save_name_origin, alpha)
cv2.imwrite(save_name_sharpen, unsharped_alpha)

# Displaying the output
disp_img(alpha)
disp_img(unsharped_alpha)

# Compositing the image with the alpha channel
background = cv2.imread('universe.jpg')
comp_img = compositing(image, alpha, background)
plt.imshow(comp_img)
plt.show()


###################################### End-to-End Test Start: ###########################################

# Get the Laplacian matting output for comparison
lap_trimap = np.array((Image.open(os.path.join(
    "data", "trimap_training_lowres", "{}.png".format(name)))))
lap_alpha, lap_result = lap(image, lap_trimap)

# MSE (Mean Squared Error) calculation and comparison
mse_original = mse_loss(alpha, image_alpha)
mse_unsharpend = mse_loss(unsharped_alpha, image_alpha)
mse_lap = mse_loss(lap_alpha, image_alpha)

# Print the MSE
print("MSE of Original Baysian matting: ", mse_original)
print("MSE of Unsharpend Baysian matting: ", mse_unsharpend)
print("MSE of Laplacian matting: ", mse_lap)

# PSNR (Peak Signal-to-Noise Ratio) calculation
MAX = 1
# Calculate PSNR
psnr_ori = 10 * np.log10(MAX**2 / mse_original)
psnr_unshp = 10 * np.log10(MAX**2 / mse_unsharpend)
psnr_lap = 10 * np.log10(MAX**2 / mse_lap)

# Print the PSNR
print("PSNR of Original Bay:", psnr_ori)
print("PSNR of Unsharped Bay:", psnr_unshp)
print("PSNR of Laplacian:", psnr_lap)

# Measures the similarity between output alpha and groundtruth alpha
ssim_ori = ssim(alpha, image_alpha,
                data_range=image_alpha.max() - image_alpha.min())
print(f"The SSIM of Original_bay is: {ssim_ori} (dB)")
ssim_unshp = ssim(unsharped_alpha, image_alpha,
                  data_range=image_alpha.max() - image_alpha.min())
print(f"The SSIM of Unsharped_bay is: {ssim_unshp} (dB)")
ssim_lap = ssim(lap_alpha, image_alpha,
                data_range=image_alpha.max() - image_alpha.min())
print(f"The SSIM of Laplacian is: {ssim_lap} (dB)")

# Time Complexity of the Baysian matting process
print("Baysian Time Complexity: ", (Bay_end - Bay_start)/2, "seconds")

# Define the function to add value labels


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], round(y[i], 6))


# Plot the bar graph of MSE
x = ['Original_Baysian', 'Unsharpend_Baysian', 'Laplacian']
y = [mse_original, mse_unsharpend, mse_lap]
plt.bar(x, y)
addlabels(x, y)
plt.xlabel('Matting Type')
plt.ylabel('MSE')
plt.ylim(0, 0.08)
plt.title('MSE of Different Matting')
plt.show()

# Plot the bar graph of PSNR
x2 = ['Original_Baysian', 'Unsharpend_Baysian', 'Laplacian']
y2 = [psnr_ori, psnr_unshp, psnr_lap]
plt.bar(x2, y2)
addlabels(x2, y2)
plt.xlabel('Matting Type')
plt.ylabel('PSNR(dB)')
# plt.ylim(0, 0.08)
plt.title('PSNR of Different Matting')
plt.show()

# Plot the bar graph of SSIM
x1 = ['Original_Baysian', 'Unsharpend_Baysian', 'Laplacian']
y1 = [ssim_ori, ssim_unshp, ssim_lap]
plt.bar(x1, y1)
addlabels(x1, y1)
plt.xlabel('Matting Type')
plt.ylabel('SSIM')
plt.ylim(0.5, 1)
plt.title('SSIM of Different Matting')
plt.show()


# Use Profiling to get a detailed report of the performance of matting function
# cProfile.run('Bayesian_Matte(image,image_trimap)')


##################################### End-to-End Test End #########################################
