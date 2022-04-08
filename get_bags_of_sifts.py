from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import cv2

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
    #                                                                          #                                                               
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    #                                                                          #
    # You will construct SIFT features here in the same way you did in         #
    # build_vocabulary (except for possibly changing the sampling rate)        #
    # and then assign each local feature to its nearest cluster center         #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''

    with open("vocab.pkl", "rb") as voc_file:
        vocab = pickle.load(voc_file)
        image_feats = np.zeros((len(image_paths), len(vocab)))

    step_sample = 2
    
    for i in range (len(image_paths)):
        img = cv2.imread(image_paths[i])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _ , descriptors = dsift(img_gray, step=[step_sample,step_sample], window_size=4, fast=True)
        descriptors =descriptors[::5]
        dist = distance.cdist(vocab, descriptors)  
        kmin = np.argmin(dist, axis = 0)
        for j in kmin:
            image_feats[i,j] += 1

    image_feats = np.matrix(image_feats)
    #print("image_feats.shape", image_feats.shape)
    
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats