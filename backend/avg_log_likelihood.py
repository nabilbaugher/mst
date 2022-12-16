import numpy as np
from test_mazes import mazes
from maze import memoize
 
@memoize
def avg_log_likelihood_decisions(decisions_list,  model ):
    ###NEEDS REWRITING
    """
    Helps compute how likely these of decisions would have been, assuming our decisions were guided
    by our model and params (higher value -> higher likelihood of decision)
    
    Represents this as the log likelihood. Averaging this over all of our decisions

    Parameters
    ----------
    decisions_list : list of tuples (Maze, Maze)
        Each tuple represents one decision our player made, moving between two nodes: (parent_maze, child_maze)
        To reach this node, our player has to have gone from its parent node, and chosen this option.
        
        This list in total represents every decision our player made.
        
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
    prob_summaries = model.choice_probs([maze for maze in mazes.values()])
    overall_probs = {}
    for prob_summary in prob_summaries:
        overall_probs.update(prob_summary)
    
    for parent_node, child_node in decisions_list: #Every decision this subject has made
        # graph = graphs[parent_node.name] #Graph for this node
        # choices = graph[parent_node]['children'] #All choices we could have made at the last step
        
        # Get the probability for each choice the parent node had
        # choices = model.choice_probs(parent_node)[parent_node]
        
        choices = overall_probs[parent_node]
        
        # We only chose the current child node
        cum_loglike += np.log(choices[child_node]) 
    
    avg_loglike = cum_loglike / len(decisions_list)
    return avg_loglike