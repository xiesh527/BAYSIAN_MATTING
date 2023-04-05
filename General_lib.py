import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# orchard_bouman_clust.py should be in the folder
from orchard_bouman_clust import clustOBouman

## Some important helper functions

# This function displays the image.
def disp_img(img):
    """
    img - input image should be a numpy array.
    """
    plt.imshow(img, cmap='gray')
    plt.show()

def compositing(img, alpha, background): 
    """
    Args:
    img - input image is an array
    alpha - input alpha channel of the image is an array
    background - background image is an array

    Returns:
    comp - composited image is an array
    """

    H = alpha.shape[0]
    W = alpha.shape[1]

    # Resizing the background image to the size of the alpha channel
    background = cv2.resize(background, (W, H))
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

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
    Args:
    shape - shape of the filter
    sigma - standard deviation of the gaussian

    Returns:
    h - gaussian filter output
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    output = np.exp(-(x*x + y*y)/(2.*sigma*sigma))
    output[output < np.finfo(output.dtype).eps*output.max()] = 0
    sumh = output.sum()
    if sumh != 0:
        output /= sumh
    return output


# To get a window where center is (x,y) that is of size (N,N)
def get_window(img,x_cor,y_cor,N=25):
    """

    Args:
    Extracts a small window of input image, around the center (x,y)
    img - image input
    x_cor,y_cor - coordinates of unknown pixel
    N - size of the window which should be odd

    Returns:
    window - window of size (N,N) around the unknown pixel as the center (x,y)
    """

    h, w, c = img.shape             # Extracting Image Dimensions
    arm = N//2                      # Arm from center to get window
    window = np.zeros((N,N,c))      

    xmin = max(0,x_cor-arm) 
    xmax = min(w,x_cor+arm+1)
    ymin = max(0,y_cor-arm)
    ymax = min(h,y_cor+arm+1)

    window[arm - (y_cor-ymin):arm +(ymax-y_cor),arm - (x_cor-xmin):arm +(xmax-x_cor)] = img[ymin:ymax,xmin:xmax]

    return window

def EM(mean_Fg, covar_Fg, mean_Bg, covar_Bg, img_ip, covar_img_ip, alpha_0, maxCount, minLike):
    """
    Args:
    mean_Fg - mean of foreground distribution
    covar_Fg - covariance of foreground distribution
    mean_Bg - mean of background distribution
    covar_Bg - covariance of background distribution
    img_ip - input image
    covar_img_ip - covariance of input image
    alpha_0 - initial value of alpha
    maxCount - maximum number of iterations
    minLike - minimum log likelihood

    Returns:
    F - foreground distribution
    B - background distribution
    alpha - alpha value
    
    """    
    I = np.eye(3)
    vals = []
    alpha = alpha_0
    count = 1
    maxlike = -np.inf
    Log_like_nm1 = -999
    Sigma_F_m1 = np.linalg.inv(covar_Fg)
    Sigma_B_m1 = np.linalg.inv(covar_Bg)
    while(1):

        Theta = np.zeros((6, 6))
        Theta[:3, :3] = Sigma_F_m1 + I * alpha * alpha /(covar_img_ip**2)
        Theta[:3,3:] = Theta[3:,:3] = I * alpha*(1-alpha)/(covar_img_ip**2)
        Theta[3:,3:] = Sigma_B_m1 + I * (1 - alpha)**2/(covar_img_ip**2)

        #Theta = np.array([[Sigma_F_m1 + I * alpha * alpha /((Sigma_C)**2), I * alpha*(1-alpha)/((Sigma_C)**2)],
                #[I * alpha * (1 - alpha)/((Sigma_C)**2), Sigma_B_m1 + I * (1 - alpha)**2/((Sigma_C)**2)]])
        Phi  = np.zeros((6, 1))
        Phi[:3] = np.reshape(Sigma_F_m1 @ mean_Fg + img_ip*(alpha) / (covar_img_ip**2),(3,1))
        Phi[3:] = np.reshape(Sigma_B_m1 @ mean_Bg + img_ip*(1-alpha) / (covar_img_ip**2),(3,1))
        #Phi = np.array([[Sigma_F_m1 * mu_F + C * alpha /((Sigma_C)**2)], [Sigma_B_m1 * mu_B + C * (1 - alpha)/((Sigma_C)**2)]])

        FB = np.linalg.solve(Theta, Phi)

        F = np.maximum(np.minimum(FB[0:3], 1), 0)
        B = np.maximum(np.minimum(FB[3:6], 1), 0)

        alpha = np.maximum(0, np.minimum(1, ((np.atleast_2d(img_ip).T-B).T @ (F-B))/np.sum((F-B)**2)))[0,0]

        F = F.T
        B = B.T

        like_C = - np.sum((np.atleast_2d(img_ip).T -alpha*F-(1-alpha)*B)**2) /(covar_img_ip**2)
        like_fg = (- ((F- np.atleast_2d(mean_Fg).T).T @ Sigma_F_m1 @ (F-np.atleast_2d(mean_Fg).T))/2)[0,0]
        like_bg = (- ((B- np.atleast_2d(mean_Bg).T).T @ Sigma_B_m1 @ (B-np.atleast_2d(mean_Bg).T))/2)[0,0]
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

def Calc(mean_Fg,covar_Fg,mean_Bg,covar_Bg,img_ip,covar_img_ip,alpha_0, maxCount, minLike):
    """
    Args:
    mean_Fg - mean of foreground distribution
    covar_Fg - covariance of foreground distribution
    mean_Bg - mean of background distribution
    covar_Bg - covariance of background distribution
    img_ip - input image
    covar_img_ip - covariance of input image
    alpha_0 - initial value of alpha
    maxCount - maximum number of iterations
    minLike - minimum log likelihood

    Returns:
    F - foreground distribution
    B - background distribution
    alpha - alpha value
    """

    vals = []

    for i in range(mean_Fg.shape[0]):
        mu_Fi = mean_Fg[i]
        Sigma_Fi = covar_Fg[i]

        for j in range(mean_Bg.shape[0]):
            mu_Bj = mean_Bg[j]
            Sigma_Bj = covar_Bg[j]

            F, B, alpha, like = EM(mu_Fi,Sigma_Fi,mu_Bj,Sigma_Bj,img_ip,covar_img_ip,alpha_0, maxCount, minLike)
            val = {'F': F, 'B': B, 'alpha': alpha, 'like': like}
            vals.append(val)

        max_like, max_index = max((v['like'], i) for i, v in enumerate(vals))
        F = vals[max_index]['F']
        B = vals[max_index]['B']
        alpha = vals[max_index]['alpha']
    return F, B, alpha



def Bayesian_Matte(img,image_trimap,N = 25,sig = 8,minNeighbours = 10):
    """
    Args:
    img - input image
    image_trimap - trimap image
    N - kernel size
    sig - standard deviation of gaussian kernel
    minNeighbours - minimum number of neighbours

    Returns:
    a_channel - alpha matte
    n_unknown - number of unknown pixels
    """
    
    #We Convert the Images to float so that we are able to play with the pixel values
    img = np.array(img,dtype = 'float')
    trimap = np.array(image_trimap, dtype = 'float')
    
    #normalise the Images to range from 0 and 1.
    img /= 255
    trimap /= 255

    #get the dimensions 
    h,w,c = img.shape
    
    kernel_std = sig
    kernel_size = N
    kernel_2d = np.outer(signal.gaussian(kernel_size, std=kernel_std), signal.gaussian(kernel_size, std=kernel_std))
    kernel_2d /= np.max(kernel_2d)
    gaussian_weights = np.tile(kernel_2d[:, :, np.newaxis], (1, 1, c))
    # print(np.shape(gaussian_weights))
    gaussian_weights = gaussian_weights[:, :, 0]
    
    fg_map = trimap == 1
    fg_actual = np.zeros((h,w,c))
    fg_actual = img * np.reshape(fg_map,(h,w,1))

    
    bg_map = trimap == 0
    bg_actual = np.zeros((h,w,c))
    bg_actual = img * np.reshape(bg_map,(h,w,1))
    
    
    unknown_map = np.logical_or(fg_map,bg_map) == False
    a_channel = np.zeros(unknown_map.shape)
    a_channel[fg_map] = 1
    a_channel[unknown_map] = np.nan

    
    n_unknown = np.sum(unknown_map)

    A,B = np.where(unknown_map == True)
    not_visited = np.vstack((A,B,np.zeros(A.shape))).T

    print("Solving Image with {} unsovled pixels... Please wait...".format(len))

    # running till all the pixels are solved.
    while(sum(not_visited[:,2]) != n_unknown):
        last_n = sum(not_visited[:,2])

        # iterating for all pixels
        for i in range(n_unknown): 
            if not_visited[i,2] == 1:
                continue
            
            #Solve if not solved
            else:
                #location of the processedpixel
                y,x = map(int,not_visited[i,:2])
                
                #take out the window that are going to be solved
                a_window = get_window(a_channel[:, :, np.newaxis], x, y, N)[:,:,0]
                
                #initialise the prior for EM algorithm (Calc Function)
                
                
                fg_window = get_window(fg_actual,x,y,N)
                fg_weights = a_window**2 * gaussian_weights 
                
                #Only want value > 0, intuitively weight should be > 0
                fg_pixels = fg_window[fg_weights > 0]
                fg_weights = fg_weights[fg_weights > 0]
                        
                
                bg_window = get_window(bg_actual,x,y,N)
                bg_weights = (1 - a_window)**2 * gaussian_weights
                bg_pixels = bg_window[bg_weights > 0]
                bg_weights = bg_weights[bg_weights > 0]
                
                # repeat this if not enough neighbors
                if len(bg_weights) < minNeighbours or len(fg_weights) < minNeighbours:
                    continue
                
                # If enough pixels, clustering to generate prior statistics for foreground and background
                mean_fg, cov_fg = clustOBouman(fg_pixels,fg_weights)
                mean_bg, cov_bg = clustOBouman(bg_pixels,bg_weights)
                mask = ~np.isnan(a_window)
                alpha_init = np.mean(a_window[mask])
                
                # We try to solve our 3 equation 7 variable problem with minimum likelihood estimation
                fg_pred,bg_pred,alpha_pred = Calc(mean_fg,cov_fg,mean_bg,cov_bg,img[y,x],0.7,alpha_init, maxCount = 50, minLike = 1e-6)
                
                
                # storing the predicted values in appropriate windows for use for later pixels.
                fg_actual[y, x, :] = fg_pred
                bg_actual[y, x, :] = bg_pred
                a_channel[y, x] = alpha_pred
                # tag this pixle as solved to continue the search/processing
                not_visited[i,2] = 1
                if(np.sum(not_visited[:,2])%1000 == 0):
                    print("Solved {} out of {}.".format(np.sum(not_visited[:,2]),len(not_visited)))

        if sum(not_visited[:,2]) == last_n:
            # ChangingWindow Size
            # Preparing the gaussian weights for window
            N += 2
            kernel_std = sig
            kernel_size = N
            kernel_2d = np.outer(signal.gaussian(kernel_size, std=kernel_std), signal.gaussian(kernel_size, std=kernel_std))
            kernel_2d /= np.max(kernel_2d)
            # Repeat the 2D kernel for each channel of the input image
            gaussian_weights = np.tile(kernel_2d[:, :, np.newaxis], (1, 1, c))
            print(np.shape(gaussian_weights))
            gaussian_weights = gaussian_weights[:, :, 0]
            print(N)

    return a_channel,n_unknown