import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import os
os.system('cls')
# orchard_bouman_clust.py should be in the folder
from orchard_bouman_clust import clustFunc

## Some important helper functions

# This function displays the image.
def show_im(img):
    """
    img - input image should be a numpy array.
    """
    plt.imshow(img, cmap='gray')
    plt.show()

def compositing(img, alpha, background): 

    H = alpha.shape[0]
    W = alpha.shape[1]

    # Resizing the background image to the size of the alpha channel
    background = cv2.resize(background, (W, H))

    # Converting the images to float
    img = img / 255
    alpha = alpha / 255
    background = background / 255

    # Reshaping the alpha channel to the size of the foreground image
    alpha = alpha.reshape((H, W, 1))
    alpha = np.broadcast_to(alpha, (H, W, 3))

    # Compositing the foreground and background images
    comp = img * (alpha) + background * (1 - alpha)

    return comp


# Provided by Matlab
def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y)/(2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# To get a window where center is (x,y) that is of size (N,N)
def get_window(img,x,y,N=25):
    """
    Extracts a small window of input image, around the center (x,y)
    img - input image
    x,y - cordinates of center
    N - size of window (N,N) {should be odd}
    """

    h, w, c = img.shape             # Extracting Image Dimensions
    
    arm = N//2                      # Arm from center to get window
    window = np.zeros((N,N,c))      

    xmin = max(0,x-arm)
    xmax = min(w,x+arm+1)
    ymin = max(0,y-arm)
    ymax = min(h,y+arm+1)

    window[arm - (y-ymin):arm +(ymax-y),arm - (x-xmin):arm +(xmax-x)] = img[ymin:ymax,xmin:xmax]

    return window

def EM(mu_F, Sigma_F, mu_B, Sigma_B, C, Sigma_C, alpha_0, maxCount, minLike):
    I = np.eye(3)

    vals = []

    alpha = alpha_0
    count = 1
    maxlike = -np.inf
    Log_like_nm1 = -999
    Sigma_F_m1 = np.linalg.inv(Sigma_F)
    Sigma_B_m1 = np.linalg.inv(Sigma_B)
    while(1):

        Theta = np.zeros((6, 6))
        Theta[:3, :3] = Sigma_F_m1 + I * alpha * alpha /((Sigma_C)**2)
        Theta[:3,3:] = Theta[3:,:3] = I * alpha*(1-alpha)/((Sigma_C)**2)
        Theta[3:,3:] = Sigma_B_m1 + I * (1 - alpha)**2/((Sigma_C)**2)

        #Theta = np.array([[Sigma_F_m1 + I * alpha * alpha /((Sigma_C)**2), I * alpha*(1-alpha)/((Sigma_C)**2)],
                #[I * alpha * (1 - alpha)/((Sigma_C)**2), Sigma_B_m1 + I * (1 - alpha)**2/((Sigma_C)**2)]])
        Phi  = np.zeros((6, 1))
        Phi[:3] = np.reshape(Sigma_F_m1 @ mu_F + C*(alpha) / ((Sigma_C)**2),(3,1))
        Phi[3:] = np.reshape(Sigma_B_m1 @ mu_B + C*(1-alpha) / ((Sigma_C)**2),(3,1))
        #Phi = np.array([[Sigma_F_m1 * mu_F + C * alpha /((Sigma_C)**2)], [Sigma_B_m1 * mu_B + C * (1 - alpha)/((Sigma_C)**2)]])

        FB = np.linalg.solve(Theta, Phi)

        F = np.maximum(np.minimum(FB[0:3], 1), 0)
        B = np.maximum(np.minimum(FB[3:6], 1), 0)

        alpha = np.maximum(0, np.minimum(1, ((np.atleast_2d(C).T-B).T @ (F-B))/np.sum((F-B)**2)))[0,0]

        F = F.T
        B = B.T

        like_C = - np.sum((np.atleast_2d(C).T -alpha*F-(1-alpha)*B)**2) /((Sigma_C)**2)
        like_fg = (- ((F- np.atleast_2d(mu_F).T).T @ Sigma_F_m1 @ (F-np.atleast_2d(mu_F).T))/2)[0,0]
        like_bg = (- ((B- np.atleast_2d(mu_B).T).T @ Sigma_B_m1 @ (B-np.atleast_2d(mu_B).T))/2)[0,0]
        like = (like_C + like_fg + like_bg)

        if like > maxlike:
            a_best = alpha
            maxLike = like
            fg_best = F.ravel()
            bg_best = B.ravel()

        if count >= maxCount or abs(like-Log_like_nm1) <= minLike:
            break

        Log_like_nm1 = like
        count = count + 1

    return F, B, alpha, like

def Calc(mu_F,Sigma_F,mu_B,Sigma_B,C,sigma_C,alpha_0, maxCount, minLike):

    vals = []

    for i in range(mu_F.shape[0]):
        mu_Fi = mu_F[i]
        Sigma_Fi = Sigma_F[i]

        for j in range(mu_B.shape[0]):
            mu_Bj = mu_B[j]
            Sigma_Bj = Sigma_B[j]

            F, B, alpha, like = EM(mu_Fi,Sigma_Fi,mu_Bj,Sigma_Bj,C,sigma_C,alpha_0, maxCount, minLike)
            val = {'F': F, 'B': B, 'alpha': alpha, 'like': like}
            vals.append(val)

        max_like, max_index = max((v['like'], i) for i, v in enumerate(vals))
        F = vals[max_index]['F']
        B = vals[max_index]['B']
        alpha = vals[max_index]['alpha']
    return F, B, alpha



def Bayesian_Matte(img,image_trimap,N = 25,sig = 8,minNeighbours = 10):
    '''
    img - input image that the user will give to perform the foreground-background mapping
    trimap - the alpha mapping that is given with foreground and background determined.
    N - Window size, determines how many pixels will be sampled around the pixel to be solved, should be always odd.
    sig - wieghts of the neighbouring pixels. less means more centered.
    minNeighbours - Neigbour pixels available to solve, should be greater than 0, else inverse wont be calculated
    '''
    
    # We Convert the Images to float so that we are able to play with the pixel values
    img = np.array(img,dtype = 'float')
    trimap = np.array(image_trimap, dtype = 'float')
    
    # Here we normalise the Images to range from 0 and 1.
    img /= 255
    trimap /= 255

    # We get the dimensions 
    h,w,c = img.shape
    
    # Preparing the gaussian weights for window
    gaussian_weights = matlab_style_gauss2d((N,N),sig)
    gaussian_weights /= np.max(gaussian_weights)

    # We seperate the foreground specified in the trimap from the main image.
    fg_map = trimap == 1
    fg_actual = np.zeros((h,w,c))
    fg_actual = img * np.reshape(fg_map,(h,w,1))

    # We seperate the background specified in the trimap from the main image. 
    bg_map = trimap == 0
    bg_actual = np.zeros((h,w,c))
    bg_actual = img * np.reshape(bg_map,(h,w,1))
    
    # Creating empty alpha channel to fill in by the program
    unknown_map = np.logical_or(fg_map,bg_map) == False
    a_channel = np.zeros(unknown_map.shape)
    a_channel[fg_map] = 1
    a_channel[unknown_map] = np.nan

    # Finding total number of unkown pixels to be calculated
    n_unknown = np.sum(unknown_map)

    # Making the datastructure for finding pixel values and saving id they have been solved yet or not.
    A,B = np.where(unknown_map == True)
    not_visited = np.vstack((A,B,np.zeros(A.shape))).T

    print("Solving Image with {} unsovled pixels... Please wait...".format(len))

    # running till all the pixels are solved.
    while(sum(not_visited[:,2]) != n_unknown):
        last_n = sum(not_visited[:,2])

        # iterating for all pixels
        for i in range(n_unknown): 
            # checking if solved or not
            if not_visited[i,2] == 1:
                continue
            
            # If not solved, we try to solve
            else:
                # We get the location of the unsolved pixel
                y,x = map(int,not_visited[i,:2])
                
                # Creating an window which states what pixels around it are solved(forground/background)
                a_window = get_window(a_channel[:, :, np.newaxis], x, y, N)[:,:,0]
                
                # Creating a window and weights of solved foreground window
                fg_window = get_window(fg_actual,x,y,N)
                fg_weights = np.reshape(a_window**2 * gaussian_weights,-1)
                values_to_keep = np.nan_to_num(fg_weights) > 0
                fg_pixels = np.reshape(fg_window,(-1,3))[values_to_keep,:]
                fg_weights = fg_weights[values_to_keep]
        
                # Creating a window and weights of solved background window
                bg_window = get_window(bg_actual,x,y,N)
                bg_weights = np.reshape((1-a_window)**2 * gaussian_weights,-1)
                values_to_keep = np.nan_to_num(bg_weights) > 0
                bg_pixels = np.reshape(bg_window,(-1,3))[values_to_keep,:]
                bg_weights = bg_weights[values_to_keep]
                
                # We come back to this pixel later if it doesnt has enough solved pixels around it.
                if len(bg_weights) < minNeighbours or len(fg_weights) < minNeighbours:
                    continue
                
                # If enough pixels, we cluster these pixels to get clustered colour centers and their covariance    matrices
                mean_fg, cov_fg = clustFunc(fg_pixels,fg_weights)
                mean_bg, cov_bg = clustFunc(bg_pixels,bg_weights)
                alpha_init = np.nanmean(a_window.ravel())
                
                # We try to solve our 3 equation 7 variable problem with minimum likelihood estimation
                fg_pred,bg_pred,alpha_pred = Calc(mean_fg,cov_fg,mean_bg,cov_bg,img[y,x],0.7,alpha_init, maxCount = 50, minLike = 1e-6)
                #fg_pred,bg_pred,alpha_pred = solve(mean_fg,cov_fg,mean_bg,cov_bg,img[y,x],0.7,alpha_init)
                
                # storing the predicted values in appropriate windows for use for later pixels.
                fg_actual[y, x] = fg_pred.ravel()
                bg_actual[y, x] = bg_pred.ravel()
                a_channel[y, x] = alpha_pred
                not_visited[i,2] = 1
                if(np.sum(not_visited[:,2])%1000 == 0):
                    print("Solved {} out of {}.".format(np.sum(not_visited[:,2]),len(not_visited)))

        if sum(not_visited[:,2]) == last_n:
            # ChangingWindow Size
            # Preparing the gaussian weights for window
            N += 2
            # sig += 1 
            gaussian_weights = matlab_style_gauss2d((N,N),sig)
            gaussian_weights /= np.max(gaussian_weights)
            print(N)

    return a_channel,n_unknown
