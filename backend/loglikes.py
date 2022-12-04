import pickle
import numpy as np
import pprint
import matplotlib.pyplot as plt
import mst_prototype as mst_prototype

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

print(TREE)

"""
Some representation of how the subjects chose to act

THIS MAY BE USEFUL REFERENCE WHEN STORING DATA FROM THE EXPERIMENTS
"""

# {subject:
#    {map:
#         {'nodes':[]}, #Which tree nodes were 
#         {'path': []}
#         }}

###Assumes that we have a pickle file
# with open(f'__experiment_1/parsed_data/subject_decisions.pickle', 'rb') as handle:
     
#     DECISIONS = pickle.load(handle)


"""
decisions_list = [(world, nid), ...]
return average loglike for sid for all decisions if world is None
if world is specified, return average loglike for decisions made in that world
"""

###Experimental data we don't have

# with open(f'__experiment_1/node_values/{model_name}/node_values_{params}.pickle', 'rb') as handle:
#     # {map: {pid: {nid: node value}}}
#     node_values = pickle.load(handle)

def loglike(params, model_name, decisions_list):
    """
    

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    model_name : TYPE
        DESCRIPTION.
    decisions_list : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    

    cum_loglike = 0

    for world, nid in decisions_list:

        pid = TREE[world][nid]['pid']

        # root node has no value
        if pid == 'NA':
            continue

        # parent node is not a decision node
        if len(TREE[world][pid]['children']) <= 1:
            continue

        cum_loglike += np.log(node_values[world][pid][nid])

    return cum_loglike / len(decisions_list)


def mle(parameters, model_name, decisions_list):
    """ maximum likelihood estimation """

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
            sid2decisions.setdefault(sid, []).extend([(world, nid) for nid in DECISIONS[sid][world]['nodes']])

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