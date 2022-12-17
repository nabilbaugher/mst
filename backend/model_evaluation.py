import pickle
import numpy as np
import pprint
import matplotlib.pyplot as plt
import random
import pandas as pd
import csv

#Maze representation
from maze import Maze, maze2tree_defunct, grid2maze
from test_mazes import graphs, mazes
#Models of human decision-making
from decisionmodel import DecisionModel, DecisionModelRange, blind_nodevalue_comb, softmax_complement, blind_nodevalue_with_memory, steps_cells_heuristic, steps_cells_heuristic_with_memory
#Converting data into our desired formats.
from data_parser import directions, decisions_to_subject_decisions, convert_data, get_csv
from avg_log_likelihood import avg_log_likelihood_decisions


pp = pprint.PrettyPrinter(compact=False, width=90)



"""
The goal of this file is to evaluate our models for decision-making. It combines three things:
    1. The maze itself,
    2. The model - how we expect humans to behave in the maze,
    3. Real data - how humans actually behave in the maze
    
    Using these three pieces, we can evaluate which models best reflect how humans actually make decisions.

========================================================

A short summary of the various functions in this file:
    
    avg_log_likelihood_decisions(decisions_list,  model ): return avg_loglike
        For a particular model, measure how well it matches human behavior using average log likelihood
        (Related to the idea of, "how likely was the model to produce this behavior")
        
        *Uses maze.maze2tree, maze.Maze, decisionsmodel.DecisionModel

    
    split_train_test_kfold(decisions_list, k=4): yield (train, test)
        Split up data into testing and training chunks to swap out for cross-validation purposes.
        
    split_train_test_rand(decisions_list, k=4): yield (train, test)
        Split up data into testing and training randomly for cross-validation purposes.
    
    model_preference(model_classes, decisions, k=4): return model_preferences
        Takes several model classes, and computes how many subjects best fit into each model class.
    
        
Reminder of the params typical format
    parent_params_comb = ('tau',)
    node_params_comb = ('gamma','beta')
"""


###Handle Cross-Validations
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



def model_preference(model_classes, subject_decisions, path, k=4):
    """
    Takes in several different model classes (DecisionModelRange objects) and evaluates how many subjects prefer each model.
    
    For each subject, we use cross-evaluation to find which model class fits best: 
    fit parameters using training data, then evaluate with testing.
    

    Returns
    -------
    model_preference : dictionary,
        Stores how successful each model class was: number of subjects who matched this class.

    """

    model_preference = {}  # {model_name: number of subjects that prefer this model}
    a_few_subjects = list(subject_decisions.keys()) # [:3] # For testing purposes
    # read in current csv, append new data if it doesn't already exist
    if path:
        current_data = pd.read_csv(path)
    else:
        current_data = {}
    
    for i, subject in enumerate(a_few_subjects):
        decisions_list = subject_decisions[subject]
        models = {}
        
        print('\nsubject', i+1, '\n')
        if path and subject in current_data['subject'].values:
            print('skipping subject', subject)
            continue
        for model_class in model_classes:
            print('\nmodel:', model_class.model_name, '\n')
            for train, test in split_train_test_rand(decisions_list, k): #Break up data into chunks
                model = model_class.fit_parameters(tuple(train)) #Get best model
                evaluation = avg_log_likelihood_decisions(tuple(test), model)
                
                model_name = model.model_name
                model_params = model.node_params + model.parent_params
                
                models[model_name] = (evaluation, model_params) #Save this model and its evaluation
        
        preferred_model = max(models, key = lambda model_name: models[model_name][0]) #Best evaluation!
        
        model_preference[subject] = (preferred_model, models[preferred_model][0], models[preferred_model][1]) #Save the best model
        if path is not None:
            with open(path, 'a') as f:
                writer = csv.writer(f)
                row_number = len(current_data) + i
                writer.writerow([row_number, subject, preferred_model, models[preferred_model][1]])

    return model_preference


# eu_model_class = DecisionModelRange(model_name= 'Expected_Utility',
#                                     evaluation_function = avg_log_likelihood_decisions,
#                                     raw_nodevalue_func = raw_nodevalue_comb,
#                                     parent_nodeprob_func = softmax,
#                                     node_params_ranges   = ((0,1,10), (0,1,10)),
#                                     parent_params_ranges = ((0,1,10),)
#                                     )

if __name__ == '__main__':
    # Models example
    # expected_utility_model = DecisionModel(model_name="Expected_Utility")
    # discounted_utility_model = DecisionModel(model_name="Discounted_Utility")
    # probability_weighted_model = DecisionModel(model_name="Probability_Weighted")


    # Define our model classes
    # eu_du_pwu = [expected_utility_model, discounted_utility_model, probability_weighted_model]

    # eu_du_pwu_model_class = DecisionModelRange(model_name= 'Expected_Utility',
    #                                     evaluation_function = avg_log_likelihood_decisions,
    #                                     raw_nodevalue_func = blind_nodevalue_comb,
    #                                     parent_nodeprob_func = softmax_complement,
    #                                     node_params_ranges   = ((0,1,5), (0,1,5)),
    #                                     parent_params_ranges = ((.2,1.8,5),)
    #                                 )
    
    # comb_memory_model_class = DecisionModelRange(model_name= 'Expected_Utility',
    #                                     evaluation_function = avg_log_likelihood_decisions,
    #                                     raw_nodevalue_func = blind_nodevalue_with_memory,
    #                                     parent_nodeprob_func = softmax_complement,
    #                                     node_params_ranges   = ((0,1,5), (0,1,5), (0,1,5), (0,1,5)),
    #                                     parent_params_ranges = ((.2,1.8,5),)
    #                                 )
    
    # testing
    # param order: beta, gamma
    expected_utility_model_class = DecisionModelRange(model_name= 'Expected_Utility',
                                                evaluation_function=avg_log_likelihood_decisions,
                                                raw_nodevalue_func=blind_nodevalue_comb,
                                                parent_nodeprob_func=softmax_complement,
                                                node_params_ranges=((1, 1, 1), (1, 1, 1)),
                                                parent_params_ranges=((10, 10, 1),)
                                            )
    
    discounted_utility_model_class = DecisionModelRange(model_name= 'Discounted_Utility',
                                                evaluation_function=avg_log_likelihood_decisions,
                                                raw_nodevalue_func=blind_nodevalue_comb,
                                                parent_nodeprob_func=softmax_complement,
                                                node_params_ranges=((0, 1, 5), (1, 1, 1)),
                                                parent_params_ranges=((10, 10, 1),)
                                            )

    probability_weighted_model_class = DecisionModelRange(model_name= 'Probability_Weighted',
                                                evaluation_function=avg_log_likelihood_decisions,
                                                raw_nodevalue_func=blind_nodevalue_comb,
                                                parent_nodeprob_func=softmax_complement,
                                                node_params_ranges=((1, 1, 1), (.25, 1.25, 3)),
                                                parent_params_ranges=((10, 10, 1),)
                                            )

    eu_du_pwu_model_class = DecisionModelRange(model_name= 'Combined_No_Memory',
                                        evaluation_function = avg_log_likelihood_decisions,
                                        raw_nodevalue_func = blind_nodevalue_comb,
                                        parent_nodeprob_func = softmax_complement,
                                        node_params_ranges   = ((0,1,3), (.5,1.5,3)),
                                        parent_params_ranges = ((10,10,1),)
                                    )
    
    comb_memory_model_class = DecisionModelRange(model_name= 'Combined_Memory',
                                        evaluation_function = avg_log_likelihood_decisions,
                                        raw_nodevalue_func = blind_nodevalue_with_memory,
                                        parent_nodeprob_func = softmax_complement,
                                        # node_params_ranges   = ((0,1,3), (.5,1.5,3), (.25,.75,3), (.25,.75,3)),
                                        node_params_ranges   = ((0,1,3), (.5,1.5,3), (.25,.75,3), (.1,.5,5)),
                                        parent_params_ranges = ((10,10,1),)
                                    )


    # get subject decisions
    file_ = get_csv("./data/prolific_data_sorted_tester_id_created_at")
    decisions = convert_data(file_)
    # print(decisions.keys())
    # subject_decisions = decisions_to_subject_decisions(decisions)
    # print(subject_decisions.keys())
    # print(subject_decisions['0e9aebe0-7972-11ed-9421-5535258a0716'])

    # get model preferences
    subject_decisions = decisions_to_subject_decisions(decisions)
    model_preference = model_preference([comb_memory_model_class], subject_decisions, './data/model_preference_more_lr_options.csv', k=4)
    # model_preference = model_preference([expected_utility_model_class, discounted_utility_model_class, probability_weighted_model_class, eu_du_pwu_model_class, comb_memory_model_class], decisions, k=4)
    # model_preference = model_preference([expected_utility_model_class, discounted_utility_model_class], decisions, k=4)
    
    print(model_preference)
    for subject in model_preference:
        print('subject', subject, 'prefers', model_preference[subject])
    
    subjects = list(model_preference.keys())
    preferred_models = []
    evaluation = []
    parameters = []
    
    for subject in subjects:
        preferred_models.append(model_preference[subject][0])
        evaluation.append(model_preference[subject][1])
        parameters.append(model_preference[subject][2])
    dict_for_df = {'subject': subjects, 'preferred_model': preferred_models, 'parameters': parameters}
    # pd.DataFrame(dict_for_df).to_csv('./data/model_preference_new.csv')
    pd.DataFrame(dict_for_df).to_csv('./data/model_preference_newer.csv')
    
    
    #Question for Zoe: each subject seems to show up multiple times. 
    #Is that when they switch from one map to the next?
    