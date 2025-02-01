import copy

import numpy as np
import torch
from torch import nn, random


# since the number of samples in all the users is same, simple averaging works
def aggregation(w):
    '''

    Function to average the updated weights of individual models to update the global model (when the number of samples is same for each individual)

    Parameters:

        w (list) : The list of state_dicts of each individual

    Returns:

        w_avg (state_dict) : The updated state_dict for global model

    '''

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def aggregation_sl_strategies(gamma, w_similar, w_outstanding, ep):
    """

    Function to average the updated weights of individual models to update the global model (when the number of samples is same for each individual)

    Parameters:

        w_outstanding (list) : A list of models of outstanding individuals
        w_similar (list) : A list of models of similar individuals
        ep : global_epoch
        gamma : The fraction of different social learning strategies

    Returns:

        w_avg (state_dict) : The updated state_dict for global model
    """
    """
    if 20 <= ep < 500:
        # Define a dynamic weight adjustment based on the difference between the numbers of models
        outstanding_to_similar_diff = len(w_outstanding) - len(w_similar)
        if outstanding_to_similar_diff >= 2:
            gamma = 0.7
        elif -2 <= outstanding_to_similar_diff < 2:
            gamma = 0.65
        else:  # outstanding_to_similar_diff < -2
            gamma = 0.6
    else:
        # Set a default weight if epoch is outside the defined range
        gamma = 0.5
    """
    # Initialize the average weights for both sets (similar and outstanding)
    
        # If either list is empty, use the other list's models directly
    if not w_outstanding:
        # If w_outstanding is empty, aggregate only from w_similar
        return aggregation(w_similar)
    elif not w_similar:
        # If w_similar is empty, aggregate only from w_outstanding
        return aggregation(w_outstanding)
    
    
    w_avg_similar = copy.deepcopy(w_similar[0])
    w_avg_outstanding = copy.deepcopy(w_outstanding[0])

    # Compute the average for the similar models
    for k in w_avg_similar.keys():
        for model in w_similar[1:]:
            w_avg_similar[k] += model[k]
        w_avg_similar[k] = torch.div(w_avg_similar[k], len(w_similar))

    # Compute the average for the outstanding models
    for k in w_avg_outstanding.keys():
        for model in w_outstanding[1:]:
            w_avg_outstanding[k] += model[k]
        w_avg_outstanding[k] = torch.div(w_avg_outstanding[k], len(w_outstanding))

    # Combine the similar and outstanding models based on the weight factor
    for k in w_avg_similar.keys():
        # Use torch.mul for element-wise multiplication for proper weighting
        w_avg_similar[k] = torch.mul(w_avg_similar[k], (1 - gamma)) + torch.mul(w_avg_outstanding[k], gamma)

    return w_avg_similar
