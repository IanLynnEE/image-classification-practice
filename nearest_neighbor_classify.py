

from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import mode

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''

    '''
    k = 5
    train_image_feats = np.array(train_image_feats)
    test_image_feats = np.array(test_image_feats)
    label = np.array(train_labels) 
    dist = distance.cdist(test_image_feats, train_image_feats, metric='cityblock')
    test_predicts = []

    for row in dist:
        kmin = np.argpartition(row, k)[:k]
        test_predicts.append( mode(label[kmin])[0][0] )
    
    '''

    '''
    k = 5
    
    train_image_feats = np.array(train_image_feats)
    test_image_feats = np.array(test_image_feats)
    test_predicts = []

    for i in train_image_feats:
        predicts = []
        distance_array = []
        for j in test_image_feats:
            distance_array.append(np.sum((i-j)**2)**(1/2))
        for j in range(k):
            min_index = np.argmin(distance_array)
            predicts.append(train_labels[min_index])
            distance_array[min_index] = 1000000000
        labels_uni, counts = np.unique(predicts,return_counts = True)
        test_predicts.append(labels_uni[np.argmax(counts)])

    '''
    k_num = 10
    train_image_feats = np.array(train_image_feats)
    test_image_feats = np.array(test_image_feats)
    test_predicts = []
    
    # TODO: Which one should be first? Logic or experiment?
    d = distance.cdist(test_image_feats, train_image_feats, metric='cosine')
    for row in d:
        # Indices of the k smallest elements in A.
        min_indices = np.argsort(row)[:k_num]
        
        ballots = [train_labels[x] for x in min_indices]
        unique_labels, counts = np.unique(ballots, return_counts=True)
        test_predicts.append(unique_labels[np.argmax(counts)])
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
    
