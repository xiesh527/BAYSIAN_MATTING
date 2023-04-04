## Importing Dependencies
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import os
os.system('cls')
# orchard_bouman_clust.py should be in the folder
from orchard_bouman_clust import clustFunc
from General_lib import *
## Some important helper functions
from calculation import *
from laplacian import *
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import time
import cProfile


name = "GT12"
save_name_origin = "ORI.png"
save_name_sharpen ="Sharpen.png"
# Reading the image and trimap
os.path.join("data","gt_training_lowres","{}.png".format(name))
image = np.array(Image.open(os.path.join("data","input_training_lowres","{}.png".format(name))))
gaussian_image = cv2.GaussianBlur(image, (25,25), 10.0)
unsharp_image = cv2.addWeighted(image, 1.5, gaussian_image, -0.5, 0)

Bay_start = time.time()
image_trimap = np.array(ImageOps.grayscale(Image.open(os.path.join("data","trimap_training_lowres","{}.png".format(name)))))
gaussian_trimap = cv2.GaussianBlur(image_trimap, (25,25), 10)
unsharp_trimap = cv2.addWeighted(image_trimap, 1.5, gaussian_trimap, -0.5, 0)
# Running the bayesian matting algorithm
#plt.imshow(image)
#plt.show()
#plt.imshow(unsharp_image)
#plt.show()
alpha,pixel_count = Bayesian_Matte(image,image_trimap) 
unsharped_alpha,unsharped_pixel_count = Bayesian_Matte(unsharp_image,unsharp_trimap) 
alpha *= 255
unsharped_alpha *= 255


image_alpha = np.array(ImageOps.grayscale(Image.open(os.path.join("data","gt_training_lowres","{}.png".format(name)))))



# Converting to int for saving
alpha_int8 = np.array(alpha,dtype = int)
Bay_end = time.time()

# Saving the alpha channel
plt.imsave("{}_alpha.png".format(name), alpha, cmap='gray')
cv2.imwrite(save_name_origin, alpha)
cv2.imwrite(save_name_sharpen, unsharped_alpha)
print(np.mean(alpha))
print(np.mean(unsharped_alpha))
show_im(alpha)
show_im(unsharped_alpha)
background = cv2.imread('universe.jpg')
comp_img = compositing(image,alpha,background)
plt.imshow(comp_img)
plt.show()


########## End-to-End Test Module: ##########


# Get the Laplacian matting output for comparison 
#Lap_start = time.time()
lap_trimap = np.array((Image.open(os.path.join("data","trimap_training_lowres","{}.png".format(name)))))
lap_alpha, lap_result = lap(image, lap_trimap)
#Lap_end = time.time()

#cm = confusion_matrix(image_alpha, alpha)

#Accuracy = metrics.accuracy_score(image_alpha, alpha)

#acc = accuracy_score(alpha.flatten(), image_alpha.flatten())

# MSE comparison
mse_original = mse_loss(alpha, image_alpha)
mse_unsharpend = mse_loss(unsharped_alpha, image_alpha)
mse_lap = mse_loss(lap_alpha, image_alpha)
#acc = accuracy(tp, tn, fp, fn) 
#prec = precision(tp, fp)
print(mse_original)
print(mse_unsharpend)
#print(acc)
print(mse_lap)

# Measure the Time Complexity of the whole Baysian matting process
print("Baysian Time Complexity: ", Bay_end - Bay_start, "seconds")
#print("Laplacian Time Complexity: ", Lap_end - Lap_start, "seconds")

# Use Profiling to get a detailed report of the performance of matting function
cProfile.run('lap(image, lap_trimap)')


# Get the confusion matrix 
alc = np.array(alpha).astype(int)

print(confusion_matrix(image_alpha, alc))

# Get the Accuracy and precision
print(Accuracy)
print(prec)


########## End-to-End Test End ##########