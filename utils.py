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
                X_shift_x = np.roll(X, l, axis=1)
                X_shift_xy = np.roll(X_shift_x, m, axis=0)
                S = np.sign(X-X_shift_xy)
                S_shift_x = np.roll(S, -l, axis=1)
                S_shift_xy = np.roll(S_shift_x, -m, axis=0)
                regularization += alpha ** (np.abs(l) + np.abs(m)) * (S - S_shift_xy)               
    return regularization            
    
# reclage orb (points d'intérêts)    
def get_transformation_orb(fixed_img, moving_img, sampling_rate):

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(moving_img, None)
    kp2, des2 = orb.detectAndCompute(fixed_img, None)

    # Affichage des points d'intérêt
    # base_keypoints = cv2.drawKeypoints(fixed_img, kp2, color=(0, 0, 255), flags=0, outImage=fixed_img)
    # test_keypoints = cv2.drawKeypoints(moving_img, kp1, color=(0, 0, 255), flags=0, outImage=moving_img)

    # Création de l'objet BFMatcher et recherche des correspondances
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Tri des correspondances selon leur distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extraction des meilleures correspondances
    num_matches = 1000
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:num_matches]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:num_matches]]).reshape(-1, 1, 2)

    # Calcul de l'homographie et recale de l'image 1 sur l'image 2
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # M1 = np.copy(M)
    M[0,2] = M[0,2] * sampling_rate
    M[1,2] = M[1,2] * sampling_rate
    return M