


from sklearn.metrics import confusion_matrix
from skimage import io
import numpy as np
import cv2
from sklearn.metrics import ConfusionMatrixDisplay

# Calculating the loss
def mse_loss(alpha, image_alpha):
    return np.sum(np.abs(alpha/255 - image_alpha/255)**2)/(alpha.shape[0]*alpha.shape[1])

def accuracy(TP, TN, FP, FN):
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    return accuracy

def precision(TP, FP):
    precision = TP/(TP+FP)
    return precision

