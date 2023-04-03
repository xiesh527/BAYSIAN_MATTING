import unittest
from bayesian_matting import *


class dim_Testing(unittest.TestCase):
    def test_input_image_dimension(self):
        a =  image.ndim
        b =  3
        self.assertEqual(a, b)

    def test_trimap_image_dimension(self):
        a =  image_trimap.ndim
        b =  2
        self.assertEqual(a, b)

    def test_alpha_image_dimension(self):
        a =  alpha.ndim
        b =  2
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()