import pickle
import numpy as np
import pprint
import matplotlib.pyplot as plt
import random


#Maze representation
from maze import Maze, maze2tree, grid2maze
#Models of human decision-making
from decisionmodel import DecisionModel, DecisionModelRange, raw_nodevalue_comb, softmax
#Converting data into our desired formats.
#import parser
from data_parser import get_csv, convert_data

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
        
    decisions_to_subject_decisions(decisions): return subject_decisions
        Convert decisions of various subjects into a more useful format.
    
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


def avg_log_likelihood_decisions(decisions_list,  model ):
    """
    Helps compute how likely these of decisions would have been, assuming our decisions were guided
    by our model and params (higher value -> higher likelihood of decision)
    
    Represents this as the log likelihood. Averaging this over all of our decisions

    Parameters
    ----------
    decisions_list : list of tuples (str, int)
        Each tuple represents one decision our player made, on one map: (map_, node)
        To reach this node, our player has to have gone from its parent node, and chosen this option.
        
        Thus, this list in total represents every decision our player made.
        
    model : DecisionModel object, representing one model for player-decisions. 
        Contains variables-
    
             model_name: str
                Contains the name of this model. Usually, the model class it is stored in.
                
             node_params: tuple of floats.
                 Contains the parameters for calculating the raw value of one node.
                 
             parent_params: tuple of floats.
                 Contains the parameters we need for calculating the probability of each child node, using values.
                 
            raw_nodevalue_func : function, optional
                A function that computes the value of a single node.
                Inputs are (maze, node, *params)
                
            parent_nodeprob_func : function, optional
                A function that computes the probability for each of our nodes.
                Inputs are (values, *params)
                
        Fully describes a single model for player action.
    

    Returns
    -------
    avg_loglike: float
        The average log likelihood of this set of decision, given our model and parameters.

    """

    cum_loglike = 0

    for maze, node in decisions_list: #Every decision this subject has made
        
        tree = maze2tree(maze) #Create a tree for this current map
        
        parent = tree[node]['pid'] #Get parent of our current node

        #Root node: no decision was made, can move on
        if parent == 'NA':
            continue

        choices = tree[parent]['children'] #All choices we could have made at the last step
        
        #If our parent had only one choice, then no decision was made: p=1, log(p)=0
        if len(choices) <= 1:
            continue
        
        #Get the node value for each choice the parent node had
        choices = model.node_values(maze, parent=parent)[parent]
        
        #We only chose the current node
        cum_loglike += np.log( choices[node] ) 
    
    avg_loglike = cum_loglike / len(decisions_list)
    
    return avg_loglike #Average out the results



def decisions_to_subject_decisions(decisions):
    """
    Convert all our player decisions into a more useful format.

    Parameters
    ----------
    decisions : dictionary of dictionaries of dictionaries
        outer dictionary: Dictionary of test subjects
            
        middle dictionary: Dictionary of mazes
            
        inner dictionary: Dictionary containing how our player moved: either nodes, or paths
        
            Stores the decisions made by every single subject we've tested.

    Returns
    -------
    subject_decisions: dictionary of lists of tuples (str, int)
        dictionary - each subject of our experiment
            list - Each choice made by our subject
                tuples - Contains (map name, node id)
        
        Contains all of the decisions made by each subject, as they move from node-to-node.
        Decisions on all different maps are put into a single list.

    """
    
    subject_decisions = {} #New format: list of all of our decisions for each subject
    
    for subject in decisions:
        subject_decisions[subject] = [] #Default
        
        for maze in decisions[subject]:
            
            nodes = decisions[subject][maze]['nodes'] #Get all of the nodes for this pair
            
            #Each decision is an event: choosing one node, on one map
            
            subject_decisions.extend( [(maze, node) for node in nodes] )  #All of the decisions for one map
            
    return subject_decisions


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






#Models example
expected_utility_model = DecisionModel(model_name="Expected_Utility")
discounted_utility_model = DecisionModel(model_name="Discounted_Utility")
probability_weighted_model = DecisionModel(model_name="Probability_Weighted")


eu_du_pwu = [expected_utility_model, discounted_utility_model, probability_weighted_model]

eu_model_class = DecisionModelRange(model_name= 'Expected_Utility',
                                    evaluation_function = avg_log_likelihood_decisions,
                                    raw_nodevalue_func = raw_nodevalue_comb,
                                    parent_nodeprob_func = softmax,
                                    node_params_ranges   = ((0,1,10), (0,1,10)),
                                    parent_params_ranges = ((0,1,10),)
                                    )

def model_preference(model_classes, decisions, k=4):
    """
    Takes in several different model classes and evaluates how many subjects prefer each model.
    
    For each subject, we use cross-evaluation to find which model class fits best: 
    fit parameters using training data, then evaluate with testing.
    

    Returns
    -------
    model_preference : dictionary,
        Stores how successful each model class was: number of subjects who matched this class.

    """

    model_preference = {}  # {model_name: number of subjects that prefer this model}
    
    subject_decisions = decisions_to_subject_decisions(decisions)
    
    for subject in decisions:
        decisions_list = subject_decisions[subject]
        
        models = {}
        
        for model_class in model_classes:
            
            for train, test in split_train_test_rand(decisions_list, k): #Break up data into chunks
            
                model = model_class.fit_parameters(train) #Get best model
                evaluation = avg_log_likelihood_decisions(test, model)
                
                model_name = model.model_name
                
                models[model_name] = evaluation #Save this model and its evaluation
        
        preferred_model = max(models, key = lambda model_name: models[model_name]) #Best evaluation!
        
        model_preference[subject] = preferred_model #Save the best model

    return model_preference


eu_model_class = DecisionModelRange(model_name= 'Expected_Utility',
                                    evaluation_function = avg_log_likelihood_decisions,
                                    raw_nodevalue_func = raw_nodevalue_comb,
                                    parent_nodeprob_func = softmax,
                                    node_params_ranges   = ((0,1,10), (0,1,10)),
                                    parent_params_ranges = ((0,1,10),)
                                    )


if __name__ == "__main__":
    pass