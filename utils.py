import cv2 as cv2
import numpy as np
<<<<<<< HEAD
=======
import itk
import matplotlib.pyplot as plt
>>>>>>> 07edad601a52f21fdaa34f0cd80a608b9834a81a

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
<<<<<<< HEAD
                regularization += alpha ** (np.abs(l) + np.abs(m)) * (np.identity(np.shape(X)[0]) - np.roll(X, -m, axis = 1) * np.roll(X, -l, axis = 0)) * np.sign(X - np.roll(X, l, axis = 0) * np.roll(X, m, axis = 1))
    return regularization            
    
=======
                X_shift_x = np.roll(X, l, axis=1)
                X_shift_xy = np.roll(X_shift_x, m, axis=0)
                S = np.sign(X-X_shift_xy)
                S_shift_x = np.roll(S, -l, axis=1)
                S_shift_xy = np.roll(S_shift_x, -m, axis=0)
                regularization += alpha ** (np.abs(l) + np.abs(m)) * (S - S_shift_xy)               
    return regularization            
   
# recalage vérité terrain cv2 
def recalage_ideal(img_ref, img_mov, M):
    img_recale = cv2.warpAffine(cv2.resize(img_mov, np.shape(img_ref), interpolation = cv2.INTER_CUBIC), M, np.shape(img_ref), flags = cv2.WARP_INVERSE_MAP)
    return img_recale

# recalage ITK
def get_transform_ITK(img_ref, img_mov, optimizer, sampling_rate):
    [H_LR, W_LR] = img_mov.shape
    initialTransform = itk.CenteredRigid2DTransform[itk.D].New() #transformation rigide
    initialParameters = initialTransform.GetParameters() #paramètres de la transformation
    initialParameters[0] = 0 #angle
    initialParameters[1] = H_LR/2 #centre de rotation
    initialParameters[2] = W_LR/2 #centre de rotation
    initialParameters[3] = 0 #tx
    initialParameters[4] = 0 #ty
    
    interpolator = itk.LinearInterpolateImageFunction[type(img_mov), itk.D].New() 
    metric = itk.MeanSquaresImageToImageMetric[type(img_ref), type(img_mov)].New()

    registration_filter = itk.ImageRegistrationMethod[type(img_ref), type(img_mov)].New() # Instance de la classe de recalage
    registration_filter.SetFixedImage(img_ref) # Image de référence
    registration_filter.SetMovingImage(img_mov) # Image à recaler
    registration_filter.SetOptimizer(optimizer) # Optimiseur
    registration_filter.SetTransform(initialTransform)  # Transformation
    registration_filter.SetInitialTransformParameters(initialParameters) #Application de la transformation initiale
    registration_filter.SetInterpolator(interpolator) # Interpolateur
    registration_filter.SetMetric(metric) # Métrique
    registration_filter.Update() # Exécution du recalage
    transform = registration_filter.GetTransform()
    
    return transform

def recalage_ITK(img_ref, img_mov, M):
    img_recale = img_ref
    return img_recale

# Entropy

def entropy(im):
    #plt.figure()
    thresh, im_thresh = cv2.threshold(im, 255, 255, cv2.THRESH_TRUNC)
    histo = np.histogram(im_thresh.ravel(), bins = 255, range=[0, np.max(im_thresh)], density=True)
    #plt.title("Histogramme " + str(im))
    H = 0
    for p in histo[0]:
        if p != 0:
            H -= p * np.log2(p)
    return H


def joint_entropy(im1, im2):
    thresh, im1_thresh = cv2.threshold(im1, 255, 255, cv2.THRESH_TRUNC)
    histo_1 = np.histogram(im1_thresh.ravel(), bins = 255, range=[0, np.max(im1_thresh)], density=True)
    thresh, im2_thresh = cv2.threshold(im2, 255, 255, cv2.THRESH_TRUNC)
    histo_2 = np.histogram(im2_thresh.ravel(), bins = 255, range=[0, np.max(im2_thresh)], density=True)
    
    histo_joint, xedges, yedges = np.histogram2d(histo_1[0], histo_2[0], bins=256)
    histo_joint = histo_joint.T # transpose
    histo_joint = histo_joint / np.sum(histo_joint)
    H = 0
    for l in histo_joint:
        for p in l:
            if p != 0:
                H -= p * np.log2(p)
    return H

def MSE(im, im_ref):
    mse = np.sum((im - im_ref)**2) / np.sum(im_ref**2)
    return mse

def PSNR(im, im_ref):
    mse = MSE(im, im_ref)
    psnr = 20 * np.log10(255/np.sqrt(mse))
    return psnr
>>>>>>> 07edad601a52f21fdaa34f0cd80a608b9834a81a
