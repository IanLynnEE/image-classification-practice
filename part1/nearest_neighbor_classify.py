from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance

def nearest_neighbor_classify(train_image_feats, train_labels, 
                                test_image_feats, k_num=5, metric='cityblock'):
    ############################################################################
    # TODO:                                                                    #
    # This function will predict the category for every test image by finding  #
    # the training image with most similar features. Instead of 1 nearest      #
    # neighbor, you can vote based on k nearest neighbors which will increase  #
    # performance (although you need to pick a reasonable value for k).        #
    ############################################################################
    ############################################################################
    # NOTE: Some useful functions                                              #
    # distance.cdist :                                                         #
    #   This function will calculate the distance between two list of features #
    #       e.g. distance.cdist(? ?)                                           #
    ############################################################################
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
        k_num :
            Integer. Number of nearest images.
        metric:
            String. Pass to distance.cdist(XA, XB, metric=metric).
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    train_image_feats = np.array(train_image_feats)
    test_image_feats = np.array(test_image_feats)
    test_predicts = []
    
    # TODO: Which one should be first? test or train?
    dist = distance.cdist(test_image_feats, train_image_feats, metric=metric)
    for row in dist:
        # Indices of the k smallest elements in A.
        min_indices = np.argsort(row)[:k_num]
        # Let the nearest to vote a label.
        ballots = [train_labels[x] for x in min_indices]
        unique_labels, counts = np.unique(ballots, return_counts=True)
        test_predicts.append(unique_labels[np.argmax(counts)])
    ############################################################################
    #                                END OF YOUR CODE                          #
    ############################################################################
    return test_predicts