from itertools import repeat
from multiprocessing import Pool
from datetime import datetime

from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import cv2

from tqdm import tqdm


def get_norm_hist(vocab, image_path, step_sample):
    img_gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    _, descriptors = dsift(img_gray, step=step_sample, fast=True)
    dist = distance.cdist(vocab, descriptors[::2])  
    kmin = np.argmin(dist, axis=0)
    hist, bin_edges = np.histogram(kmin, bins=len(vocab))
    return hist / sum(hist)

def get_bags_of_sifts(image_paths, step_sample=8):
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
        step_sample : Pass to dsift. 
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    print('In get_bags_of_sifts, step_sample =', step_sample)
    print('In get_bags_of_sifts, calculating histogram...', end='')
    start_time = datetime.now()
    pool = Pool(6)
    image_feats = pool.starmap(get_norm_hist, 
        zip(repeat(vocab), image_paths, repeat(step_sample)))
    end_time = datetime.now()
    print('Done! Duration:', (end_time - start_time))
    ############################################################################
    #                                END OF YOUR CODE                          #
    ############################################################################
    return np.matrix(image_feats)
