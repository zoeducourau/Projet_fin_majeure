import cv2 as cv2
import numpy as np

import scipy.sparse as sp

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
    
# ADMM
def prox_gammal1Z(Z, lambda_, gamma):
    return (Z > lambda_ * gamma) * (Z - lambda_ * gamma) + (Z < - lambda_ * gamma) * (Z + lambda_ * gamma)



# Matrice D de derivation
def get_D(n, m):

    S = n * m
    siz = (n, m)

    Dx = sp.lil_matrix((S, S))
    Dx.setdiag(-np.ones(S))
    Dx.setdiag(np.ones(S), siz[1])
    Dx = Dx / 2
    Dx[-siz[1]:, :] = 0

    Dy = sp.lil_matrix((S, S))
    Dy.setdiag(-np.ones(S))
    Dy.setdiag(np.ones(S), 1)
    Dy = Dy / 2
    Dy[siz[1]-1::siz[1], :] = 0

    D = sp.vstack([Dx, Dy])
    
    return D


def regularization_ADMM(X, Z, D):
    D_X_vectorized = D.dot(X.flatten())
    D_X = D_X_vectorized.reshape(X.shape[0] * 2, X.shape[1])
    diff = Z - D_X
    regularization_vectorized = np.transpose(D).dot(diff.flatten())
    regularization = regularization_vectorized.reshape(X.shape[0], X.shape[1])
    
    return regularization


def dX_bilateral_TV(X, Z, alpha, P):
    output  = np.zeros(np.shape(X))   
    for l in range(-P, P + 1):
        for m in range(0, P + 1):
            if m + l >= 0 :
                X_shift_i = np.roll(X, l, axis=1)
                X_shift_ij = np.roll(X_shift_i, m, axis=0)
                S = X - X_shift_ij
                diff = Z - S
                S_shift_i = np.roll(diff, -l, axis=1)
                S_shift_ij = np.roll(S_shift_i, -m, axis=0)
                output += alpha ** (np.abs(l) + np.abs(m)) * (diff - S_shift_ij)
    return output

def G_D_prox_bilateral_TV(X, Z, alpha, P, lambda_, gamma):
    output = np.zeros(np.shape(Z))
    for l in range(-P, P + 1):
        for m in range(0, P + 1):
            if m + l >= 0 :
                X_shift_i = np.roll(X, l, axis=1)
                X_shift_ij = np.roll(X_shift_i, m, axis=0)
                S = X - X_shift_ij
                output += alpha ** (np.abs(l) + np.abs(m)) * prox_gammal1Z(Z - gamma * (Z - S), lambda_, gamma)
    return output