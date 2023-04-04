import unittest
from General_lib import get_window
import cv2
import numpy as np

image = cv2.imread('data/input_training_lowres/GT01.png')
image_trimap = cv2.imread('data/trimap_training_lowres/GT01.png')
alpha = cv2.imread('GT12_alpha.png')
comp_img = cv2.imread('GT12_compsite_img.png')


class dim_Testing(unittest.TestCase):
    def test_input_image_dimension(self):
        a =  image.ndim
        b =  3
        self.assertEqual(a, b)

    def test_trimap_image_dimension(self):
        a =  image_trimap.ndim
        b =  3
        self.assertEqual(a, b)

    def test_alpha_image_dimension(self):
        a =  alpha.ndim
        b =  3
        self.assertEqual(a, b)

    def test_comp_image_dimension(self):
        a =  comp_img.ndim
        b =  3
        self.assertEqual(a, b)

class value_Testing(unittest.TestCase):
    def test_trimap_image_min_value(self):
        a =  image_trimap.min()
        b =  0
        self.assertEqual(a, b)

    def test_trimap_image_max_value(self):
        a =  image_trimap.max()
        b =  255
        self.assertEqual(a, b)

class test_window(unittest.TestCase):
    def test_window_size(self):
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

class test_EM(unittest.TestCase):
    def test_EM(self):
        pass

if __name__ == '__main__':
    unittest.main()