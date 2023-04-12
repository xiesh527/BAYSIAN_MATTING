import numpy as np
import cv2
from table import my_dict

# GT = "GT01"
def segment_prediction(image, threshold, row_start, row_end, column_start, column_end):
    # threshold = my_dict[GT]["threshold"]  
    # row_start = my_dict[GT]["row_start"]  
    # row_end = my_dict[GT]["row_end"]  
    # column_start = my_dict[GT]["column_start"]  
    # column_end = my_dict[GT]["column_end"] 

    linear_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y = linear_img[:, :, 0]
    u = linear_img[:, :, 1]
    v = linear_img[:, :, 2]

    # Cropping the image
    bg_crop = linear_img[row_start:row_end, column_start:column_end]

    # Extracting the y, u, v channels for the background
    By = bg_crop[:, :, 0]
    Bu = bg_crop[:, :, 1]
    Bv = bg_crop[:, :, 2]

    # Calculating the mean of y channel, u channel and v channel
    By_mean = np.mean(By)
    Bu_mean = np.mean(Bu)
    Bv_mean = np.mean(Bv)

    # Calculating the variance of y channel, u channel and v channel
    By_var = np.var(By)
    Bu_var = np.var(Bu)
    Bv_var = np.var(Bv)

    # Calculating the threshold for y channel, u channel and v channel
    Bg_threshold_y = (y - By_mean)**2 / (2 * By_var) < threshold
    Bg_threshold_u = (u - Bu_mean)**2 / (2 * Bu_var) < threshold
    Bg_threshold_v = (v - Bv_mean)**2 / (2 * Bv_var) < threshold

    # Adding the theshold of three channels
    Bg_threshold = Bg_threshold_y + Bg_threshold_u + Bg_threshold_v

    # Calculating the alpha channel
    alpha_inv = Bg_threshold.astype(np.int32)
    alpha = 1 - alpha_inv
    return alpha