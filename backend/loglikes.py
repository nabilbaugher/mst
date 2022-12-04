import pickle
import numpy as np
import pprint
import matplotlib.pyplot as plt
import mst_prototype as mst_prototype

from mst_prototype import map2tree, node_values
from mst_prototype import raw_nodevalue_comb

pp = pprint.PrettyPrinter(compact=False, width=90)

import models

# with open(f'__experiment_1/parsed_data/tree.pickle', 'rb') as handle:
#     TREE = pickle.load(handle)
map_1 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
         (3, 3, 3, 3, 0, 3, 0, 3, 3),
         (3, 3, 3, 3, 0, 3, 0, 3, 3),
         (3, 5, 6, 6, 6, 6, 6, 6, 3),
         (3, 6, 3, 3, 3, 3, 3, 6, 3),
         (3, 6, 6, 6, 6, 6, 6, 6, 3),
         (3, 3, 0, 0, 3, 3, 3, 3, 3),
         (3, 3, 3, 3, 3, 3, 3, 3, 3),)

###Test tree: map_1
TREE= mst_prototype.map2tree(map_1)



"""
Some representation used by the original writers of this code

THIS MAY BE USEFUL REFERENCE WHEN STORING DATA FROM THE EXPERIMENTS
"""

# DECISIONS =
# {subject:
#    {map:
#         {'nodes':[]}, 
#         {'path': []}
#         }}

###Experimental data we don't have
# with open(f'__experiment_1/parsed_data/subject_decisions.pickle', 'rb') as handle:
     
#     DECISIONS = pickle.load(handle)

# with open(f'__experiment_1/node_values/{model_name}/node_values_{params}.pickle', 'rb') as handle:
#     # {map: {pid: {nid: node value}}}
#     node_values = pickle.load(handle)




def avg_log_likelihood_decisions(decisions_list,  params, raw_nodevalue_func=raw_nodevalue_comb ):
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
        
    params : tuple of three floats.
        parameters to help calculate values. 
        
        If using raw_nodevalue_func=raw_nodevalue_comb, 
        Contains our parameters (tau, gamma, beta).
                 
    raw_nodevalue_func : function, optional
        A function that computes the value of a single node.
        Configurable, so we can try out different functions/parameters for computing node values.
    

    Returns
    -------
    float
        The average log likelihood of this set of decision, given our model and parameters.

    """

    cum_loglike = 0

    for map_, node in decisions_list: #Every decision this subject has made
        
        tree = map2tree(map_) #Create a tree for this current map
        
        parent = tree[node]['pid'] #Get parent of our current node

        #Root node: no decision was made, can move on
        if parent == 'NA':
            continue

        choices = tree[parent]['children'] #All choices we could have made at the last step
        
        #If our parent had only one choice, then no decision was made: p=1, log(p)=0
        if len(choices) <= 1:
            continue
        
        #Get the node value for each choice the parent node had
        choices = node_values(map_, params, raw_nodevalue_func=raw_nodevalue_comb, 
                              parent = parent)[parent]
        
        #We only chose the current node
        cum_loglike += np.log( choices[node] ) 
    
    return cum_loglike / len(decisions_list) #Average out the results




def best_parameters_from_decisions(decisions_list, parameters, raw_nodevalue_func=raw_nodevalue_comb):
    """
    For a given model, with multiple options for parameters, determine which parameters 
    best fit the player's decisions.

    Parameters
    ----------
    decisions_list : list of tuples (str, int)
        Each tuple represents one decision our player made, on one map: (map_, node)
        To reach this node, our player has to have gone from its parent node, and chosen this option.
        
        Thus, this list in total represents every decision our player made.
    
    parameters : list of tuples of three floats.
        Each tuple represents one "setting" for our model.
        
        parameters to help calculate values. 
        
            If using raw_nodevalue_func=raw_nodevalue_comb, 
            Contains our parameters (tau, gamma, beta).
        
    raw_nodevalue_func : function, optional
        A function that computes the value of a single node.
        Configurable, so we can try out different functions/parameters for computing node values.
    
    Returns
    -------
    tuple of (float, tuples(float, float, float) )
        First float is the log likelihood of our best parameters.
        The tuple is the collection of best parameters.

    """
    
    model_results = []
    
    for params in parameters: #Try every option for our model
        log_likelihood = avg_log_likelihood_decisions(decisions_list,  params, 
                                                      raw_nodevalue_func=raw_nodevalue_comb )
        
        model_results.append( (log_likelihood, params) ) #Store the results
        
    return max(model_results) #Pick the model with the highest log likelihood! 


# DECISIONS =
# {subject:
#    {map:
#         {'nodes':[]}, 
#         {'path': []}
#         }}




def convert_decisions_to_subject_decisions(decisions):
    """
    Convert all our player decisions into a more useful format.

    Parameters
    ----------
    Parameters
    ----------
    decisions : dictionary of dictionaries of dictionaries
        outer dictionary: Dictionary of test subjects
            
        middle dictionary: Dictionary of maps
            
        inner dictionary: Dictionary containing how our player moved: either nodes, or paths
        
            Stores the decisions made by every single subject we've tested.

    Returns
    -------
    subject_decisions: dictionary of lists of tuples (str, int)
        Dictionary - each subject of our experiment
            List - Each choice made by our subject
                Tuples - Contains (map name, node id)
        
        Contains all of the decisions made by each subject, as they move from node-to-node.
        Decisions on all different maps are put into a single list.

    """
    
    # {subject: [ (map1, node1), 
    #             (map2, node2) ] }
    
    subject_decisions = {} #New format: list of all of our decisions for each subject
    
    for subject in decisions:
        subject_decisions[subject] = [] #Default
        
        for map_ in decisions[subject]:
            
            nodes = decisions[subject][map_]['nodes'] #Get all of the nodes for this pair
            
            #Each decision is an event: choosing one node, on one map
            
            subject_decisions.extend( [(map_, node) for node in nodes] )  #All of the decisions for one map
            
    return subject_decisions


eu_du_pwu = [('Expected_Utility', raw_nodevalue_comb, (1,1,1)),
             ('Discounted_Utillity', raw_nodevalue_comb, (1,1,1)),
             ('Probability_Weighted', raw_nodevalue_comb, (1,1,1))]

def parameter_fitting_per_subject(decisions, models):
    """
    For each subject, find the best parameters for our model.

    Parameters
    ----------
    decisions : dictionary of dictionaries of dictionaries
        outer dictionary: Dictionary of test subjects
            
        middle dictionary: Dictionary of maps
            
        inner dictionary: Dictionary containing how our player moved: either nodes, or paths
        
            Stores the decisions made by every single subject we've tested.
            

    Returns
    -------
    None.

    """
    
    


# if __name__ == "__main__":
#     sid = 'S99991343'
#     parameters = [(tau, 1, 1) for tau in models.TAUS]
#     model_name = 'Expected_Utility'

#     parameters = [(round(tau, 3), round(gamma, 3), 1) for tau in models.TAUS for gamma in models.GAMMAS]
#     model_name = 'Discounted_Utility'

#     parameters = [(round(tau, 3), 1, round(gamma, 3)) for tau in models.TAUS for gamma in models.BETAS]
#     model_name = 'Probability_Weighted_Utility'

#     # loglike(sid, parameters[0], model_name)
#     # mle(sid, parameters, model_name)
#     model_fitting(parameters, model_name)