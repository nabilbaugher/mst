import pickle
import numpy as np
import pprint
import matplotlib.pyplot as plt
import mst_prototype as mst_prototype

from mst_prototype import map2tree, node_values
from mst_prototype import raw_nodevalue_comb, softmax

pp = pprint.PrettyPrinter(compact=False, width=90)


"""
A short summary of the various functions in this file:
    
    avg_log_likelihood_decisions(decisions_list,  model ): return avg_loglike
        For a particular model, measure how well it matches human behavior using average log likelihood
        (Related to the idea of, "how likely was the model to produce this behavior")
        
        *Uses mst_prototype.map2tree, mst_prototype.node_values
        
    decisions_to_subject_decisions(decisions): return subject_decisions
        Convert decisions of various subjects into a more useful format.
    
    best_model_calculation(decisions_list, models): return best_model
        Calculates which model best fits our observed decisions.
        
        *Uses avg_log_likelihood_decisions
        
    best_model_each_subject(decisions, models): return best_model_per_subject
        Finds the best-fitting model for each of our subjects individually.
        
        *Uses decisions_to_subject_decisions, best_model_calculation
    
    best_model_all_subjects(decisions, models): return best_model
        Finds the best-fitting single model for all of our subjective collectively
        
        *Uses decisions_to_subject_decisions, best_model_calculation
    
"""

###Experimental data we don't have

# with open(f'__experiment_1/parsed_data/tree.pickle', 'rb') as handle:
#     TREE = pickle.load(handle)
# with open(f'__experiment_1/parsed_data/subject_decisions.pickle', 'rb') as handle:
#     DECISIONS = pickle.load(handle)
# with open(f'__experiment_1/node_values/{model_name}/node_values_{params}.pickle', 'rb') as handle:
#     # {map: {pid: {nid: node value}}}
#     node_values = pickle.load(handle)


#Models example
eu_du_pwu = [('Expected_Utility',     (1,1), (1,), raw_nodevalue_comb, softmax, ),
             ('Discounted_Utillity',  (1,1), (1,), raw_nodevalue_comb, softmax, ),
             ('Probability_Weighted', (1,1), (1,), raw_nodevalue_comb, softmax, )]

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
        
    model : tuple of the form (str, function, function, tuple of floats, tuple of floats)
            str is the name of the model
            
            first tuple has the parameters for node value
            second tuple has the parameters for node probability
            
            first function is used to compute node value
            second function is used to compute node probability
            
        Fully describes a single model for player action.
    

    Returns
    -------
    avg_loglike: float
        The average log likelihood of this set of decision, given our model and parameters.

    """
    
    #Unpack our model
    (_, node_params, parent_params, raw_nodevalue_func, parent_nodeprob_func) = model

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
        choices = node_values(map_,  node_params, parent_params,
                        raw_nodevalue_func,  parent_nodeprob_func, #Optional
                        parent=parent)[parent]
        
        #We only chose the current node
        cum_loglike += np.log( choices[node] ) 
    
    avg_loglike = cum_loglike / len(decisions_list)
    
    return avg_loglike #Average out the results


# DECISIONS =
# {subject:
#    {map:
#         {'nodes':[]}, 
#         {'path': []}
#         }}




def decisions_to_subject_decisions(decisions):
    """
    Convert all our player decisions into a more useful format.

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
        dictionary - each subject of our experiment
            list - Each choice made by our subject
                tuples - Contains (map name, node id)
        
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


# DECISIONS =
# {subject:
#    {map:
#         {'nodes':[]}, 
#         {'path': []}
#         }}

def best_model_calculation(decisions_list, models):
    """
    Given a list of decisions taken, give the best-fitting model for this behavior.

    Parameters
    ----------
    decisions_list : list of tuples (str, int)
        Each tuple represents one decision our player made, on one map: (map_, node)
        To reach this node, our player has to have gone from its parent node, and chosen this option.
        
        Thus, this list in total represents every decision our player made.
        
    models : list of tuples 
        Each tuple of the form (str, function, function, tuple of floats, tuple of floats)
            str is the name of the model
            
            first tuple has the parameters for node value
            second tuple has the parameters for node probability
            
            first function is used to compute node value
            second function is used to compute node probability
            
        Each tuple represents a model we want to try out.

    Returns
    -------
    best_model : tuple of the form (str, function, function, tuple of floats, tuple of floats)
                str is the name of the model
                
                first tuple has the parameters for node value
                second tuple has the parameters for node probability
                
                first function is used to compute node value
                second function is used to compute node probability
        
        The best model for our entire experimental sample.

    """
    
    ll_models = {} #Log likelihood
    
    for model in models:
        ll = avg_log_likelihood_decisions(decisions_list,  model ) #Get log likelihood
        
        ll_models[model] = ll #Save each ll value
        
    best_model = max( ll_models, key= ll_models[model] ) #Find the best model
    
    return best_model
    
    

def best_model_each_subject(decisions, models):
    """
    For each subject, find the best model that matches their behavior.

    Parameters
    ----------
    decisions : dictionary of dictionaries of dictionaries
        outer dictionary: Dictionary of test subjects
            
        middle dictionary: Dictionary of maps
            
        inner dictionary: Dictionary containing how our player moved: either nodes, or paths
        
            Stores the decisions made by every single subject we've tested.
            
    models : list of tuples 
        Each tuple of the form (str, function, function, tuple of floats, tuple of floats)
            str is the name of the model
            
            first tuple has the parameters for node value
            second tuple has the parameters for node probability
            
            first function is used to compute node value
            second function is used to compute node probability
            
        Each tuple represents a model we want to try out.
        
    Returns
    -------
    best_model_per_subject : 
        Dictionary of tuples - keys are subjects, vals are models
        
            tuple of the form (str, function, function, tuple of floats, tuple of floats)
                str is the name of the model
                
                first tuple has the parameters for node value
                second tuple has the parameters for node probability
                
                first function is used to compute node value
                second function is used to compute node probability
        
        A dictionary where each subject (key) is matched to their best-fitting model (value)
    
    
    """
    
    subject_decisions = decisions_to_subject_decisions(decisions)
    
    best_model_per_subject = {}
    
    for subject in subject_decisions:
        decisions_list = subject_decisions[subject]
        
        best_model = best_model_calculation(decisions_list, models)
        
        best_model_per_subject[subject] = best_model #Assign
        
    return best_model_per_subject
        
        

def best_model_all_subjects(decisions, models):
    """
    For all subjects, find the best model that matches their overall behavior.

    Parameters
    ----------
    decisions : dictionary of dictionaries of dictionaries
        outer dictionary: Dictionary of test subjects
            
        middle dictionary: Dictionary of maps
            
        inner dictionary: Dictionary containing how our player moved: either nodes, or paths
        
            Stores the decisions made by every single subject we've tested.
            
    models : list of tuples 
        Each tuple of the form (str, function, function, tuple of floats, tuple of floats)
            str is the name of the model
            
            first tuple has the parameters for node value
            second tuple has the parameters for node probability
            
            first function is used to compute node value
            second function is used to compute node probability
            
        Each tuple represents a model we want to try out.
        
    Returns
    -------
    best_model:tuple of the form (str, function, function, tuple of floats, tuple of floats)
                str is the name of the model
                
                first tuple has the parameters for node value
                second tuple has the parameters for node probability
                
                first function is used to compute node value
                second function is used to compute node probability
        
        The best model for our entire experimental sample.
    """    
    
    subject_decisions = decisions_to_subject_decisions(decisions)
    
    decisions_list = []
    
    for subject in subject_decisions: #Add up the decisions made by every subject
        decisions_list += subject_decisions[subject]
        
    best_model = best_model_calculation(decisions_list, models)
    
    return best_model
            

def fit_parameters(decisions_list, model_name, node_params_ranges, parent_params_ranges, 
                   raw_nodevalue_func, parent_nodeprob_func):
    """
    For a given model type (functions unchanged), fit parameters to a pattern of human decisions.

    Parameters
    ----------
    decisions_list : list of tuples (str, int)
        Each tuple represents one decision our player made, on one map: (map_, node)
        To reach this node, our player has to have gone from its parent node, and chosen this option.
        
        Thus, this list in total represents every decision our player made.
        
    model_name : str
        The name of the model we are fitting.
        
    node_params_ranges : tuple of tuples (float, float, int)
        Outer tuple - each inner tuple represents a different parameter.
        Each inner tuple represents the range we want to vary a parameter over: (min, max, number_of_samples)
        
        Allows us to try out different param values in given ranges.
        For node value parameters.
        
    node_params_ranges : tuple of tuples (float, float, int)
        Outer tuple - each inner tuple represents a different parameter.
        Each inner tuple represents the range we want to vary a parameter over: (min, max, number_of_samples)
        
        Allows us to try out different param values in given ranges.
        For parent probability parameters.
        
    raw_nodevalue_func : function, optional
        A function that computes the value of a single node.
        
    parent_nodeprob_func : function, optional
        A function that computes the probability for each of our nodes.

    Returns
    -------
    None.

    """
    
    ##This function is incomplete!
    

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

# map_1 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
#          (3, 3, 3, 3, 0, 3, 0, 3, 3),
#          (3, 3, 3, 3, 0, 3, 0, 3, 3),
#          (3, 5, 6, 6, 6, 6, 6, 6, 3),
#          (3, 6, 3, 3, 3, 3, 3, 6, 3),
#          (3, 6, 6, 6, 6, 6, 6, 6, 3),
#          (3, 3, 0, 0, 3, 3, 3, 3, 3),
#          (3, 3, 3, 3, 3, 3, 3, 3, 3),)

# ###Test tree: map_1
# TREE= mst_prototype.map2tree(map_1)