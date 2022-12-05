import pickle
import numpy as np
import pprint
import matplotlib.pyplot as plt
import random

import loglikes

from mst_prototype import raw_nodevalue_comb, softmax

pp = pprint.PrettyPrinter(compact=False, width=90)

#Data we don't have
# with open(f'__experiment_1/parsed_data/tree.pickle', 'rb') as handle:
#     TREE = pickle.load(handle)

# with open(f'__experiment_1/parsed_data/subject_decisions.pickle', 'rb') as handle:
#     # {sid: {world: {'nodes': [], 'path': []}}}
#     DECISIONS = pickle.load(handle)


def split_train_test_kfold(decisions_list, k=4):
    """
    Splits our data (decisions) into k evenly sized, disjoint chunks ("splits")
    To be used for cross-evaluation.

    Parameters
    ----------
    decisions_list : list of tuples (str, int)
        Each tuple represents one decision our player made, on one map: (map_, node)
        To reach this node, our player has to have gone from its parent node, and chosen this option.
        
        Thus, this list in total represents every decision our player made.
        
    k : int, optional
        Number of chunks we want to split our data into.

    Yields
    ------
    train : list of tuples (str, int)
        (k-1) chunks of the data, to be used for training a model.
    test : TYPE
        1 chunk of data, to be used for testing the model.
        
    Yields every combination of training and testing for our chunks.

    """

    n = len(decisions_list) // k  # length of each split
    splits = [decisions_list[i * n: (i + 1) * n] for i in range(k)] #Split data into chunks

    for i in range(k): #For each chunk, make a dataset where that chunk is testing, and the rest are training
        test = splits[i] #Main chunk
        train = sum([splits[j] for j in range(k) if j != i]) #Other chunks
        

        yield train, test


def split_train_test_rand(decisions_list, k=4):
    """
    Similar to split_train_test_kfold, but we randomly select which data points go in the testing chunk.
    To be used for cross-evaluation.

    Parameters
    ----------
    decisions_list : list of tuples (str, int)
        Each tuple represents one decision our player made, on one map: (map_, node)
        To reach this node, our player has to have gone from its parent node, and chosen this option.
        
        Thus, this list in total represents every decision our player made.
        
    k : int, optional
        Number of chunks we want to split our data into.

    Yields
    ------
    train : list of tuples (str, int)
        (k-1) chunks of the data, to be used for training a model.
    test : TYPE
        1 chunk of data, to be used for testing the model.
        
    Yields k different randomly-generated training/testing datasets.

    """

    for _ in range(k):
        test = random.sample(decisions_list, k=len(decisions_list) // k)
        train = [d for d in decisions_list if d not in test]

        yield train, test


# model2parameters = {
#     'Expected_Utility': [(tau, 1, 1) for tau in models.TAUS],
#     'Discounted_Utility': [(round(tau, 3), round(gamma, 3), 1) for tau in models.TAUS for gamma in models.GAMMAS],
#     'Probability_Weighted_Utility': [(round(tau, 3), 1, round(beta, 3)) for tau in models.TAUS for beta in
#                                      models.BETAS]
# }

#{model_name: {'best_parameters': ((),()),
#              'num_subjects_preferred':}}

"""
return model preference summary
{model_name: number of subjects that prefer this model}
"""

#Models example
expected_utility_model = {'model_name': 'Expected_Utility',
                          'node_params': (1,1), #gamma, beta
                          'parent_params':(1,), #tau,
                          'raw_nodevalue_func': raw_nodevalue_comb,
                          'parent_nodeprob_func': softmax}

discounted_utility_model = {'model_name': 'Discounted_Utility',
                            'node_params': (1,1), #gamma, beta
                            'parent_params':(1,), #tau,
                            'raw_nodevalue_func': raw_nodevalue_comb,
                            'parent_nodeprob_func': softmax}


probability_weighted_model = {'model_name': 'Probability_Weighted',
                              'node_params': (1,1), #gamma, beta
                              'parent_params':(1,), #tau,
                              'raw_nodevalue_func': raw_nodevalue_comb,
                              'parent_nodeprob_func': softmax}

def model_preference():
    """
    Takes in several different types of models and evaluates how many subjects prefer each model.
    For each subject, we use cross-evaluation to find which model type fits best: fit parameters using training data, then evaluate with testing.
    

    Returns
    -------
    model_preference : dictionary
        Stores how successful each model class was:

    """
    



    model_preference = {}  # {model_name: number of subjects that prefer this model}
    
    for subject in 
    
    for sid in DECISIONS:
        decisions_list = []
        max_avg_loglike = float('-inf')

        # collect all decisions made by subject sid
        for world in DECISIONS[sid]:
            decisions_list.extend((world, nid) for nid in DECISIONS[sid][world]['nodes'])

        for model_name, parameters in model2parameters.items():

            avg_test_loglike, k = 0, 4

            for train, test in split_train_test_rand(decisions_list, k):
                max_loglike, mle_params = loglikes.mle(parameters, model_name, train)
                avg_test_loglike += loglikes.loglike(mle_params, model_name, test) / k

            if avg_test_loglike > max_avg_loglike:
                max_avg_loglike = avg_test_loglike
                best_model = model_name

        model_preference[best_model] = model_preference.get(best_model, 0) + 1

    pp.pprint(model_preference)
    return model_preference




if __name__ == "__main__":
    model_preference()