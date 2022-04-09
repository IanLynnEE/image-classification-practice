from PIL import Image
import numpy as np
import cv2

def crop_square(img):
    h, w = img.shape
    min_hw = min(h, w)
    y = h // 2 - min_hw // 2; x = w // 2 - min_hw // 2
    return img[y:y+min_hw, x:x+min_hw]

def get_tiny_images(image_paths, interpolation=cv2.INTER_AREA):
    ###########################################################################
    # TODO:                                                                   #
    # To build a tiny image feature, simply resize the original image to      # 
    # a very small square resolution, e.g. 16x16.                             #
    # You can either ignore their aspect ratio or crop the center square      #
    # portion out of each image before resizing the images to square.         #
    # Making the tiny images zero mean and unit length (normalizing them)     # 
    # will increase performance modestly.                                     #  
    ###########################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny_images: (N, d) matrix of resized and then vectorized tiny images.
        E.g. if the images are resized to 16x16, d would equal 256.
    '''
    tiny_images = []
    for i in range(len(image_paths)):
        img = cv2.cvtColor(cv2.imread(image_paths[i]), cv2.COLOR_BGR2GRAY)
        square = crop_square(img)
        tiny = cv2.resize(square, (16, 16), interpolation=interpolation)
        tiny1D = tiny.flatten()
        norm = (tiny1D - np.mean(tiny1D)) / np.std(tiny1D)
        tiny_images.append(norm)
    ###########################################################################
    #                                END OF YOUR CODE                         #
    ###########################################################################
    return np.matrix(tiny_images)
