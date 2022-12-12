
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import pprint
from collections import deque

from maze import Maze, maze2tree, grid2maze, memoize

from test_mazes import mazes, trees, maze2tree_dict
from data_parser import decisions_to_subject_decisions

"""
The goal of this file is to create the DecisionModel and DecisionModelRange classes. Why do we need this classes?

Our focus is the Maze Search Task. This task is used partly to try to figure out how humans make decisions, 
in a controlled environment (i.e., the maze).

To figure out this decision-making process, we create some of our "best guesses" as to how humans make decisions.
These guesses are *models* that mathematically represents one explanation for human behavior.

We try to fit our models to human decision-making, to improve them. By seeing which model holds up
best on new data, we can see which models more accurately reflect how humans think.

Each DecisionModel object is one of the models we're trying out, with its own parameters.

DecisionModelRange describes a whole group of similar models, which we can call a "model class".
(Here, the use of the word 'class' is separate from the Python meaning of class.)

These classes are built-in with functions that make it easier to find the best
model for human behavior.

========================================================

A short summary of the various functions in this file:
    
    softmax(values, tau): return softvals
        Applies softmax to our values, with the parameter tau to make terms more/less similar.
    
    weight(p, beta): returns p_weighted
        Re-weights probabilities, using parameter beta so that 
        low probabilities are overestimated, and high probabilities are underestimated.
        
    raw_nodevalue_comb(map_, node, gamma=1, beta=1): return value
        Calculates value of a particular node, using gamma and beta parameters.
        Gamma makes future events less valuable: value is scaled down by gamma for each timestep.
        Beta is the same as in weight.
        
        *Uses weight
        
    generate_combinations(array): return combinations
        Takes one array storing several different arrays, and finds every way to combine one element from each array.
        
        
class DecisionModel():
    Represents one particular model for human decision-making in maze task.
    
    model_copy(self): return DecisionModel
        Return a copy of the model.
        
    update_params(self, node_params=None, parent_params=None): return None
        Update the parameters of this model.
        
    node_values(self, maze, parent=None): return values
        Calculate the value of each node in the maze. 
        Only the children of one parent, if that parent is specified.
        
        *Uses maze2tree
        

class DecisionModelRange(DecisionModel):
    Represents one class of model for human decision-making in maze task - parameters given over a range.
    
    gen_model(self, node_params, parent_params): return DecisionModel
        Create a model with the specified parameters.
        
        *Uses self.model_copy, self.update_params
        
    gen_node_param_range(self): return node_params_arrays
        Takes the stored range, and produces all of the parameter values we need for the node function.
        
    gen_parent_param_range(self): return node_params_arrays
        Takes the stored range, and produces all of the parameter values we need for the parent function.
        
    gen_models(self): return models
        Creates all of the models our parameter ranges specify.
        
        *Uses self.gen_node_param_range, self.gen_parent_param_range
        
    fit_parameters(self, decisions_list): return model
        Based on our evaluation function, and the decisions made by our player(s), determine which model parameters fit best.
        
        *Uses self.gen_models
"""

def softmax(values, tau):
    """
    Applies the softmax function to our inputs. Tau allows us to emphasize or de-emphasize the
    difference between our values.

    Parameters
    ----------
    values : List of ints
        Set of values we want to take the softmax over.
        
    tau :    float
        Controls how "noisy" our softmax operation is: larger tau increases "noise",
        making our output terms more similar to each other.
        
        tau = 1: No effect. Softmax operates normally.
        tau > 1: Output terms more similar - closer to a uniform distribution.
        tau < 1: Output terms more different - emphasize larger values, downscale small values

    Returns
    -------
    list
        Returns the output of a softmax over our values.

    """

    numer = [np.exp(-v * ( 1 /tau)) for v in values]
    denom = sum(numer)
    return [ n /denom for n in numer]

""" probability weighting function: convert probability p to weight """

def weight(p, beta):
    """
    Re-weights probabilities, modelling on the human tendency 
    to overestimate small probabilities, and underestimate large probabilities.
    
    Parameters
    ----------
    p : float in real number range [0,1]
        Probability of an event.
        
    beta : float in real number range [0, inf]
        Represents the amount we follow the human pattern described above -
        bringing extreme values near 0 or 1 closer to the middle 0.5.
        
        beta = 1: No effect.
        beta > 1: Shows the effect desired: extremes move closer to 0.5.
        beta < 1: Opposite effect: extremes move further away from  0.5.

    Returns
    -------
    float
        Result of weighting: a probability in range [0,1]
    
    
    """
    
    return np.exp( -1 * (-np.log(p) )**beta )




###Rather than creating multiple different models, 
###the combined value model is used to generalize all three: EU, DU, PWU

def compute_value_iteration(maze):
    pass

def compute_exit_probability(parent_hidden, child_hidden, beta):
    """
    Get the probability that we find the exit at the child node.
    
    Helper function for raw_nodevalue_comb.
    """
    revealed =set(parent_hidden) - set(child_hidden) #Revealed is how many are no longer hidden
    
    prob_of_exit = len(revealed)/len(parent_hidden) #Probability that we see the right exit
    
    weighted_prob = weight(prob_of_exit, beta)   #Adjust based on human bias
    weighted_comp = weight(1-prob_of_exit, beta) #Complement probability
    
    return weighted_prob, weighted_comp
    

@memoize #Hopefully saves on time cost to compute?
def raw_nodevalue_comb(maze, gamma=1, beta=1):
    ###NEEDS REWRITING
    """
    Get value of this node (this path through our maze), 
    according to our parameters, before applying softmax and thus tau.
    
    If parameters are set to 1, then value of this node is the expected distance from this node to the exit.

    Parameters
    ----------
    maze : Maze object, our maze to navigate. Stores a maze.map object:
        
        map: tuple of tuples of ints - represents a grid in the shape (maze.nrows, maze.ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
               Our "maze" the player is moving through.
        
    gamma : float in real number range [0,1],  optional
        The "discount factor". Reflects the idea that we care less about rewards in the future.
        For every timestep in the future, we scale down the reward by a factor of gamma.
        
        Thus, larger gamma means we care more about future rewards. 
        Smaller gamma means we care less about future rewards.
        
    beta : float in real number range [0, inf], optional
        Represents the amount we follow the human pattern described above -
        bringing extreme values near 0 or 1 closer to the middle 0.5.
        
        beta = 1: No effect.
        beta > 1: Shows the effect desired: extremes move closer to 0.5.
        beta < 1: Opposite effect: extremes move further away from  0.5.

    Returns
    -------
    value : float
        Represents how "valuable" we think this particular path is, based on 
        what we know about possible futures, and our parameters.
        
        This does not take softmax into account.

    """

    tree = maze2tree_dict(maze)
    
    #Get hidden squares
    parent_hidden = maze.get_hidden()
    parent_node = parent_hidden
    
    #Values for each child we could choose
    child_losses = {}
    
    #All children
    children = tree[parent_node]["children"]
    
    for child_node in children:
        child_maze   = children[child_node]['map'] 
        child_hidden = child_node #Node identified by hiddens
        
        ###Here, we get the probability of finding an exit at this node
        
        weighted_prob, weighted_comp = compute_exit_probability(parent_hidden, child_hidden, beta)
        
        ###Distance from parent to child
        
        child_path = children[child_node]['path'] #What is the known path to this node?
        parent_to_child = len(child_path) #Our guaranteed loss: how far to we walk to make this inspection?
        
        ###Average distance to exit: assuming the exit is at the child node
        
        player_pos     = np.array(child_maze.pos) 
        possible_exits = np.array(list(revealed))
        
        #Since we're on a grid, we take *manhattan distance*
        
        delta_to_exit   = np.abs( possible_exits - player_pos) #Difference in position, take abs
        dists_to_exit   = np.sum( delta_to_exit, axis=1) #Add x and y coords for manhattan dist
        
        child_to_exit   = np.mean(dists_to_exit ) #Average out each exit
        
        ###Average distance to exit: assuming exit is not at the child node, and is instead later.
        child_to_exit_later = raw_nodevalue_comb(child_maze, gamma, beta)
        
        #No matter what, we will get this loss: we have to walk to the child to find out whether we win.
        necessary_loss = parent_to_child
        #Loss assuming we find the exit
        success_loss = weighted_prob * child_to_exit
        #Loss assuming we don't find the exit at this child node
        failure_loss = gamma * weighted_comp *  child_to_exit_later
        
        #Expected loss: each loss has been scaled by its probability of occurring
        child_loss = necessary_loss + success_loss + failure_loss
        child_losses[child_node] = child_loss #All losses
        
    #We pick the child that incurs the least loss
    true_loss = min(child_losses, key = lambda child: child_losses[child])

    return true_loss


  
def generate_combinations(array):
    """
    For n different sub-arrays, find every way to draw one element from each array.
    
    For example, [a,b] and [1,2] make [ [a,1], [a, 2], [b,1], [b,2]
    
    Parameters
    ----------
    array: list of lists 
        Our outer list contains multiple arrays of elements that we want to use in our combination.
        
        We will be pulling one element from each inner list to get our combinations.
    
    Returns
    -------
    combinations: list of lists
        Each list is a unique combination of elements: one from each of our starting arrays.

    """
    combinations = []
    
    if len(array)==0: #Base case: no arrays, no combinations.
        return [[]]
    
    
    for elem in array[0]: #Otherwise, recurse: we iterate over our first array
        
        future_combinations = generate_combinations(array[1:]) #Iterate over the next array, then the next one...
        
        for comb in future_combinations: #We've got all of our results: combine them with our current element
            
            combinations.append( [elem] + comb ) #Assemble full combination
            
    return combinations



#############
# MAIN RELEVANT CLASS
#############

class DecisionModel:
    """
        Represents one model we could use to represent the choices a user makes while navigating a maze.
    """
    
    def __init__(self, model_name, 
                 node_params=(1,1), parent_params=(1,), 
                 raw_nodevalue_func=raw_nodevalue_comb, parent_nodeprob_func=softmax):
        
        """
        model_name: str
            Contains the name of this model. Usually, the model class it is stored in.
        """
        self.model_name = model_name
        
        """
        node_params: tuple of floats.
            Contains the parameters for calculating the raw value of one node.
        """
        self.node_params = node_params
        """
        parent_params: tuple of floats.
            Contains the parameters we need for calculating the probability of each child node, using values.
        """
        self.parent_params = parent_params
        """
        raw_nodevalue_func : function, optional
            A function that computes the value of a single node.
            Inputs are (maze, node, *params)
        """
        self.raw_nodevalue_func = raw_nodevalue_func
        """
        parent_nodeprob_func : function, optional
            A function that computes the probability for each of our nodes.
            Inputs are (values, *params)
        """
        self.parent_nodeprob_func = parent_nodeprob_func 
    
    def model_copy(self):
        """
        Copy this model.
        """
        return DecisionModel(self.model_name, self.node_params, self.parent_params, 
                             self.raw_nodevalue_func, self.parent_nodeprob_func)
    
    def update_params(self, node_params=None, parent_params=None):
        """
            Update our parameters based on what we are given.
            
            View __init__ for description.
        """
        if node_params:
            self.node_params = node_params
        if parent_params:
            self.parent_params = parent_params
            
    
    def node_values(self, maze, parent=None):
        ###NEEDS REWRITING
        """
        Returns the value of every possible path (node) for the entire map.
        

        Parameters
        ----------
        maze : Maze object, our maze to navigate. Stores a maze.map object:
            
            map: tuple of tuples of ints - represents a grid in the shape (maze.nrows, maze.ncols)
                   -tuple of tuples    has length = nrows, 
                   -each tuple of ints has length = ncols.
                   
                   Our "maze" the player is moving through.

        parent : int, optional
            If given this parameter, only find the node values which are the children of this node.
            This can be used to compare different choices.

        Returns
        -------
        values_summary : 
            Dictionary of dictionaries
                Outer dictionary: key - each node in our tree excluding the root node.
                                  val - dict of each child and their val.
                
                Inner dictionary: key - the child node we're focused on.
                                  val - the value of that child node.

        """

        values_summary = {} # {node: {child_node: value, child_node: value, ...}}
        tree = maze2tree(maze)
            

        for node in tree: 
            
            if parent!=None: #Exclusively focus on this node, rather than others
                if node != parent:
                    continue

            children = tree[node]['children']

            # If node doesn't allow a choice, ignore it.
            if len(children) <= 1:
                continue
            
            #Inner dictionary
            values_summary[node] = {}
            
            #Calulate value of each child, pre-softmax
            raw_values = [ self.raw_nodevalue_func(maze, child_node, *self.node_params) 
                          for child_node in children ]
            
            #Apply whatever function you want to these raw values
            values = self.parent_nodeprob_func(raw_values, *self.parent_params) 
            
            values_summary[node] = {child_node: val for child_node,val in zip(children, values)}
        
        return values_summary
    




class DecisionModelRange(DecisionModel):
    """A model class, with a range of possible parameters."""
    
    def __init__(self, model_name, 
                 node_params_ranges, parent_params_ranges, evaluation_function,
                 raw_nodevalue_func=raw_nodevalue_comb, parent_nodeprob_func=softmax,):
        
        super().__init__(self, model_name, raw_nodevalue_func=raw_nodevalue_comb, parent_nodeprob_func=softmax)
        #Preserve usual DecisionModel stuff
        
        
        #Add ranges
        """
        node_params_ranges: list of tuple (float, float, int)
            Each list element (a tuple) represents a different parameter.
            The parameter range is specified by three things: (start, end, samples)
        """
        self.node_params_ranges = node_params_ranges
        """
        parent_params_ranges: list of tuple (float, float, int)
            Each list element (a tuple) represents a different parameter.
            The parameter range is specified by three things: (start, end, samples)
        """
        self.parent_params_ranges = parent_params_ranges
        
        """
        evaluation_function : function
            Function that helps us determine which model is best.
            Assumed to take in (decision_list, model). 
        """
        self.evaluation_function = evaluation_function
        
    def gen_model(self, node_params, parent_params):
        """
        Create new model using parameters!
        """
        
        new_model = self.model_copy() #Create copy of this model!
        new_model.update_params(node_params, parent_params)
        
        return new_model
    
    def gen_node_param_range(self):
        """
        Give all values for each parameter, for node values

        Returns
        -------
        list of lists
            Each inner list represents a different parameter.
            The floats contained within are values for that parameter.

        """
        return [ list( np.linspace(*param_range) )
                for param_range in self.node_params_ranges ]
    
    def gen_parent_param_range(self):
        """
        Give all values for each parameter, for node values

        Returns
        -------
        list of lists
            Each inner list represents a different parameter.
            The floats contained within are values for that parameter.

        """
        return [ list( np.linspace(*param_range) )
                for param_range in self.parent_params_ranges ]
    
    def gen_models(self):
        """
        Generate every possible model based on the parameter ranges specified.

        Returns
        -------
        models : list of DecisionModel objects.
            All of our models.

        """
        node_params_arrays =   self.gen_node_param_range()
        
        parent_params_arrays = self.gen_parent_param_range()
        
        all_node_params = generate_combinations(node_params_arrays)
        all_parent_params = generate_combinations(parent_params_arrays)
        
        models = []
        
        #Generate all of the models we want
        
        for node_params in all_node_params: #Every parameter combo creates its own model
            for parent_params in all_parent_params:
        
                new_model = self.gen_model(node_params, parent_params)
                
                models.append(new_model)
                
        return models
                
    def fit_parameters(self, decisions_list):
        """
        Select the set of parameters that gives this function the best outcome for our evaluation function. 

        Parameters
        ----------
        decisions_list : list of tuples (str, int)
            Each tuple represents one decision our player made, on one map: (map_, node)
            To reach this node, our player has to have gone from its parent node, and chosen this option.
            
            Thus, this list in total represents every decision our player made.


        Returns
        -------
        model: DecisionModel object
            Model which performed best on the given task.

        """
        
        models = self.gen_models()
        
        performance = {}
        
        for index, model in enumerate(models):
            
            evaluation = self.evaluation_function(decisions_list,  model ) #Get model performance
            
            performance[index] = evaluation #Save each value
            
        best_index = max( performance, key= performance[index] ) #Find the best model
        
        return models[best_index]
            

