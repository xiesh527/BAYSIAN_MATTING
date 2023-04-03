from sklearn.metrics import confusion_matrix
from skimage import io
import numpy as np
import cv2
from sklearn.metrics import ConfusionMatrixDisplay
from General_lib import *
save_name_origin = "ORI.png"
save_name_sharpen ="Sharpen.png"

ori_img = cv2.imread(save_name_origin)
sharpen_img = cv2.imread(save_name_sharpen)

show_im(ori_img)
show_im(sharpen_img)