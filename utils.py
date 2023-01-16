import cv2 as cv2
import numpy as np

# down-sampling
def decimation(input_img, ds_rate):
    down_sampled_img = np.zeros([int(np.shape(input_img)[0]/ds_rate), int(np.shape(input_img)[1]/ds_rate)])
    for i in range(np.shape(input_img)[0]):
        for j in range(np.shape(input_img)[1]):
            i_ds = int(i/ds_rate)
            j_ds = int(j/ds_rate)
            if i % ds_rate == 0 and j % ds_rate == 0 :
                down_sampled_img[i_ds, j_ds] = input_img[i, j]
        
    return down_sampled_img

# up-sampling
def interpolation(input_img, us_rate):
    up_sampled_img = np.zeros([np.shape(input_img)[0]*us_rate, np.shape(input_img)[1]*us_rate])
    for i in range(np.shape(input_img)[0]):
        for j in range(np.shape(input_img)[1]):
            up_sampled_img[i*us_rate, j*us_rate] = input_img[i, j]                  
    return up_sampled_img

# gradient G1
def G1(X, Y, M, kernel_size, sampling_rate):
    difference = decimation(cv2.GaussianBlur(cv2.warpAffine(X, M, np.shape(X)), (kernel_size, kernel_size), 0), sampling_rate) - Y
    gradient = cv2.warpAffine(cv2.GaussianBlur(interpolation(np.sign(difference), sampling_rate), (kernel_size, kernel_size), 0), M, np.shape(X), flags = cv2.WARP_INVERSE_MAP)
    return gradient

# gradient G2
def G2(X, Y, M, kernel_size, sampling_rate):
    difference = decimation(cv2.GaussianBlur(cv2.warpAffine(X, M, np.shape(X)), (kernel_size, kernel_size), 0), sampling_rate) - Y
    gradient = cv2.warpAffine(cv2.GaussianBlur(interpolation(difference, sampling_rate), (kernel_size, kernel_size), 0), M, np.shape(X), flags = cv2.WARP_INVERSE_MAP)
    return gradient


# regularization Tikhonov
def tikhonov(X, gamma):
    regularization = 2*cv2.filter2D(cv2.filter2D(X, ddepth = -1 , kernel = gamma), ddepth = -1 , kernel = gamma)
    return regularization

def TV(X, gamma):
    regularization = np.sign(2*cv2.filter2D(cv2.filter2D(X, ddepth = -1 , kernel = gamma), ddepth = -1 , kernel = gamma))
    return regularization

# bilateral Total Variation
def bilateral_TV(X, alpha, P) :
    regularization = np.zeros(np.shape(X))
    for l in range(-P, P + 1):
        for m in range(0, P + 1):
            if m + l >= 0:
                regularization += alpha ** (np.abs(l) + np.abs(m)) * (np.identity(np.shape(X)[0]) - np.roll(X, -m, axis = 1) * np.roll(X, -l, axis = 0)) * np.sign(X - np.roll(X, l, axis = 0) * np.roll(X, m, axis = 1))
    return regularization            
    