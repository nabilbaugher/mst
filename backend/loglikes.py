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


def mle(parameters, model_name, decisions_list):
    """
    Determines which model is most probable, based on negative log likelihood minimization
    """

    max_loglike = float('-inf')

    for params in parameters:

        avg_loglike = loglike(params, model_name, decisions_list)

        if avg_loglike > max_loglike:
            max_loglike = avg_loglike
            mle_params = params

    return max_loglike, mle_params


def model_fitting(parameters, model_name):
    # {sid: list of decision nodes}
    sid2decisions = {}

    # {sid: (max_loglike, mle_params)}
    sid2mle = {}

    for sid in DECISIONS:
        for world in DECISIONS[sid]:
            sid2decisions.setdefault(sid, []).extend([(world, nid) 
                                                      for nid in DECISIONS[sid][world]['nodes']])

        sid2mle[sid] = mle(parameters, model_name, sid2decisions[sid])

    _, axs = plt.subplots(1, 3)
    axs = axs.flat

    axs[0].hist([max_ll for max_ll, _ in sid2mle.values()], edgecolor='white')
    axs[1].hist([params[0] for _, params in sid2mle.values()], edgecolor='white')
    axs[2].hist([params[2] for _, params in sid2mle.values()], edgecolor='white')

    plt.show()


if __name__ == "__main__":
    sid = 'S99991343'
    parameters = [(tau, 1, 1) for tau in models.TAUS]
    model_name = 'Expected_Utility'

    parameters = [(round(tau, 3), round(gamma, 3), 1) for tau in models.TAUS for gamma in models.GAMMAS]
    model_name = 'Discounted_Utility'

    parameters = [(round(tau, 3), 1, round(gamma, 3)) for tau in models.TAUS for gamma in models.BETAS]
    model_name = 'Probability_Weighted_Utility'

    # loglike(sid, parameters[0], model_name)
    # mle(sid, parameters, model_name)
    model_fitting(parameters, model_name)