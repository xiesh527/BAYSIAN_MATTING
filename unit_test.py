# Importing libraries
import unittest
import cv2
import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os

# Importing the functions from personal libraries
from General_lib import get_window
from orchard_bouman_clust import calculate_weighted_mean, calculate_weighted_covariance
from table import my_dict
from alpha_prediction import segment_prediction

# Reading the image 
# image = cv2.imread('data/input_training_lowres/GT01.png')
# cv2.imread('data/input_training_lowres/GT01.png')
# print(image.shape)

# # False image resized
image = cv2.imread('data/input_training_lowres/GT02.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (800, 497))


image_trimap = np.array(ImageOps.grayscale(Image.open(
    os.path.join("data", "trimap_training_lowres", "{}.png".format('GT01')))))
alpha = cv2.imread('GT12_alpha.png')
comp_img = cv2.imread('GT12_compsite_img.png')

# Defining the parameters
GT = "GT01"
threshold = my_dict[GT]["threshold"]  
row_start = my_dict[GT]["row_start"]  
row_end = my_dict[GT]["row_end"]  
column_start = my_dict[GT]["column_start"]  
column_end = my_dict[GT]["column_end"] 
alpha_GT = my_dict[GT]["alpha"]

#  Testing if image and trimap are same
class test_image_trimap(unittest.TestCase):
    def test_alpha(self):
        GT = "GT01"
        white_calculated = my_dict[GT]["alpha"]
        alpha_pred1 = segment_prediction(image, threshold, row_start, row_end, column_start, column_end)
        white_predicted1 = np.sum(alpha_pred1 == 1)
        self.assertEqual(white_calculated, white_predicted1)

    def test_trimap(self):
        white_pixel1 = my_dict[GT]["trimap"]
        white_pixel2 = np.sum(image_trimap == 255)
        self.assertEqual(white_pixel1, white_pixel2)

    # def test_alpha_trimap(self):
    #     a = white_predicted1
    #     b = white_pixel2
    #     self.assertEqual(a, b)

#  Checking the value of trimap is between 0 and 255
class value_Testing(unittest.TestCase):
    def test_trimap_image_min_value(self):
        a =  image_trimap.min()
        b =  0
        self.assertEqual(a, b)

    def test_trimap_image_max_value(self):
        a =  image_trimap.max()
        b =  255
        self.assertEqual(a, b)

# Checking if the unknown pixels are cornered or not
class test_window(unittest.TestCase):
    def test_odd_window_size(self):
        w_size = get_window(image,100, 100, 25)
        size = 25
        self.assertEqual(w_size.shape[0], size)
        self.assertEqual(w_size.shape[1], size)
    
    def test_first_corner_window(self):
        w_size = get_window(image, 0, 0, 25)
        self.assertEqual(w_size.shape[0], 25)
        self.assertEqual(w_size.shape[1], 25)

    def test_last_pixel_window(self):
        w_size = get_window(image, 500, 500, 25)
        self.assertEqual(w_size.shape[0], 25)
        self.assertEqual(w_size.shape[1], 25)

# Checking if the weighted mean and weighted covariance are correct
class test_Orchard_Bouman(unittest.TestCase):
    def test_calculate_weighted_mean(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        wg = np.array([0.4, 1.0, 0.6])
        mean = calculate_weighted_mean(X, wg)
        expected_mean = np.array([3.2, 4.2])
        assert_array_almost_equal(mean, expected_mean)

    def test_calculate_weighted_covariance(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        wg = np.array([0.1, 0.3, 0.6])
        mean = np.array([4, 5, 6])
        covar = calculate_weighted_covariance(X, wg, mean)
        expected_covar = np.array([[6.30001, 6.3, 6.3],
                                    [6.3, 6.30001, 6.3],
                                    [6.3, 6.3, 6.30001]])
        np.testing.assert_array_almost_equal(covar, expected_covar, decimal=5)

if __name__ == '__main__':
    unittest.main()