import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import pprint
from collections import deque

from maze import Maze, maze2tree_defunct, grid2maze, memoize

from test_mazes import mazes, graphs #maze2graph_dict
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
        
    choice_probs(self, maze): return probs_summary
        Calculate the probability of choosing a child node, given you are in a particular parent node.
        
        *Uses test_mazes.graphs
        

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

###############
# Code begins #
###############

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


# probability weighting function: convert probability p to weight
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
    return np.exp( -1 * (-err_log(p) )**beta )


def err_log(x):
    if x == 0:
        return -float('inf')
    return np.log(x)


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


def compute_exit_distance(revealed, child_pos):
    """
    Assuming that we reveal an exit at our current node, how far away is that exit from the node on average?
    
    Helper function for raw_nodevalue_comb.
    """
    #Get the positions
    player_pos     = np.array(child_pos) 
    possible_exits = np.array(list(revealed))
    
    #Since we're on a grid, we take *manhattan distance*
    delta_to_exit   = np.abs( possible_exits - player_pos) #Difference in position, take abs
    dists_to_exit   = np.sum( delta_to_exit, axis=1) #Add x and y coords for manhattan dist
    
    return np.mean(dists_to_exit ) #Average out each exit

@memoize #Hopefully saves on time cost to compute?
def blind_nodevalue_comb(maze, gamma=1, beta=1):
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
        
    graph = graphs[maze.name]
    
    #Data about parent
    parent_node = maze
    parent_hidden = maze.get_hidden()
    
    #Values for each child we could choose
    child_losses = {}
    
    #All children
    children = graph[parent_node]["children"]
    
    for child_maze, path_to_child in children.items():
        child_hidden = child_maze.get_hidden()
        
        ###Here, we get the probability of finding an exit at this node
        weighted_prob, weighted_comp = compute_exit_probability(parent_hidden, child_hidden, beta)
        
        #Distance from parent to child
        parent_to_child = len(path_to_child) #Our guaranteed loss: how far to we walk to make this inspection?
        
        ###Average distance to exit: assuming the exit is at the child node
        revealed = set(parent_hidden) - set(child_hidden)
        
        child_pos = child_maze.pos
        
        child_to_exit = compute_exit_distance(revealed, child_pos)
        
        ###Average distance to exit: assuming exit is not at the child node, and is instead later.
        child_to_exit_later = blind_nodevalue_comb(child_maze, gamma, beta)
        
        #No matter what, we will get this loss: we have to walk to the child to find out whether we win.
        necessary_loss = parent_to_child
        #Loss assuming we find the exit
        success_loss = weighted_prob * child_to_exit
        #Loss assuming we don't find the exit at this child node
        failure_loss = gamma * weighted_comp *  child_to_exit_later
        
        #Expected loss: each loss has been scaled by its probability of occurring
        child_loss = necessary_loss + success_loss + failure_loss
        child_losses[child_maze] = child_loss #All losses
    
    if children == {}:
        return 0
    
    #We pick the child that incurs the least loss
    true_loss = min(child_losses.values())  
    
    return true_loss


def get_distances_to_exit(maze):
    """
    Get the distance from each node to the exit.
    """
    nrows = maze.nrows
    ncols = maze.ncols
    result = [[float('inf') for _ in range(nrows)] for _ in range(ncols)]
    result[maze.exit[0]][maze.exit[1]] = 0
    non_walls = set(maze.black) | set(maze.path) | set([maze.exit]) | set([maze.start])
    
    # bfs
    q = deque()
    q.append((maze.exit[0], maze.exit[1], 0))
    visited = set()
    while q:
        row, col, distance = q.popleft()
        if (row, col) in visited:
            continue
        if (row, col) not in non_walls:
            continue
        
        result[row][col] = distance
        q.append((row + 1, col, distance + 1))
        q.append((row - 1, col, distance + 1))
        q.append((row, col + 1, distance + 1))
        q.append((row, col - 1, distance + 1))
        visited.add((row, col))
    return result
    


@memoize
def value_iteration(prev_mazes, learning_rate=.5):
    """
    For each maze, calculate the value of each node based on the position of the exit.
    Calculated as min distance to exit from each node.
    Essentially "seen_nodevalue_comb"
    Returns matrix of values for each node.
    """
    nrows = prev_mazes[0].nrows
    ncols = prev_mazes[0].ncols
    result = [[0 for _ in range(nrows)] for _ in range(ncols)]
    if len(prev_mazes) > 1:
        result = value_iteration(prev_mazes[:-1], learning_rate)
    
    distances = get_distances_to_exit(prev_mazes[-1])
    for i in range(nrows):
        for j in range(ncols):
            result[i][j] = (1 - learning_rate) * result[i][j] + learning_rate * distances[i][j]
    return result


def get_non_wall_filter(matrix, filter_, position):
    """
    returns a filter that has zero values for all positions that are walls and all values add to 1.
    """
    nrows = len(matrix)
    ncols = len(matrix[0])
    new_filter_not_normalized = [[0 for _ in range(len(filter_))] for _ in range(len(filter_))]
    for i in range(len(filter_)):
        for j in range(len(filter_)):
            row = position[0] - len(filter_) // 2 + i
            col = position[1] - len(filter_) // 2 + j
            if row < 0 or row >= nrows or col < 0 or col >= ncols:
                continue
            if matrix[row][col] != float('inf'):
                new_filter_not_normalized[i][j] = filter_[i][j]
    new_filter = [[0 for _ in range(len(filter_))] for _ in range(len(filter_))]
    new_sum = sum([sum(row) for row in new_filter_not_normalized])
    for i in range(len(filter_)):
        for j in range(len(filter_)):
            new_filter[i][j] = new_filter_not_normalized[i][j] / new_sum
    return new_filter


def apply_filter(matrix, filter_, position):
    """
    Apply a filter to a matrix, centered at a given position.
    """
    filter_ = get_non_wall_filter(matrix, filter_, position)
    nrows = len(matrix)
    ncols = len(matrix[0])
    filter_nrows = len(filter_)
    filter_ncols = len(filter_[0])
    result = 0
    for i in range(filter_nrows):
        for j in range(filter_ncols):
            row = position[0] + i - filter_nrows // 2
            col = position[1] + j - filter_ncols // 2
            if row < 0 or row >= nrows or col < 0 or col >= ncols:
                continue
            if matrix[row][col] == float('inf'):
                continue
            result += matrix[row][col] * filter_[i][j]
    return result

@memoize #Hopefully saves on time cost to compute?
def blind_nodevalue_with_memory(child_maze, prev_mazes, discount_factor=1, bad_estimation_factor=1, memory_weight=.5, learning_rate=.5):
    """
    Q_blind = blind nodevalue no memory
    Q_seen = matrix of nodevalues based on past maps exits, no info about current map
    Q_past = nodevalue based on past maps exits, and current map layout
    Q_mem = weighted average of Q_seen and Q_past
    """
    Q_blind = blind_nodevalue_comb(child_maze, discount_factor, bad_estimation_factor)
    if len(prev_mazes) == 0:
        return Q_blind
    Q_seen = value_iteration(prev_mazes, learning_rate)
    
    # 5x5 filter for now
    # averages over 5x5 area with more weight on center
    filter_ = [[.01, .02, .04, .02, .01],
               [.02, .04, .08, .04, .02],
               [.04, .08, .16, .08, .04],
               [.02, .04, .08, .04, .02],
               [.01, .02, .04, .02, .01]]
    
    Q_past = apply_filter(Q_seen, filter_, child_maze.pos)
        
    Q_mem = memory_weight * Q_past + (1 - memory_weight) * Q_blind
    return Q_mem    
  

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
    if len(array)==0: #Base case: no arrays, no combinations.
        return [[]]
    
    combinations = []
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
                 raw_nodevalue_func=blind_nodevalue_comb, parent_nodeprob_func=softmax):
        
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
            
    
    def choice_probs(self, mazes):
        """
        Returns the probability of every choice we could make in the graph.

        Parameters
        ----------
        maze : Maze object, our maze to navigate. Stores a maze.map object:
            map: tuple of tuples of ints - represents a grid in the shape (maze.nrows, maze.ncols)
                   -tuple of tuples    has length = nrows, 
                   -each tuple of ints has length = ncols.
                   
                   Our "maze" the player is moving through.

        parent : boolean, optional
            If given this parameter, only find the choice probs which are the children of this node.
            This can be used to compare different choices.

        Returns
        -------
        probs_summary : 
            Dictionary of dictionaries
                Outer dictionary: key - each node in our graph ("pre-choice" state)
                                  val - dict of each child and their val.
                
                Inner dictionary: key - the child node we're focused on.
                                  val - the probability of our model choosing that child node,
                                        given our parent node.
        """
        result = []
        for i, maze in enumerate(mazes):
            probs_summary = {} # {node: {child_node: value, child_node: value, ...}}
            graph = graphs[maze.name]
            prev_mazes = tuple(mazes[:i])
            
            for parent_node in graph: 
                children = graph[parent_node]['children']
                
                #Inner dictionary
                probs_summary[parent_node] = {}
                raw_values = []
                
                ###Note: this way of doing the loss is still a little weird
                ###There's also some repeat logic from the other function: maybe we can do something about that?
                for child_maze, path in children.items():
                    parent_to_child = len(path) #Distance to child
                    #Average value of this child: average distance to exit
                    child_to_exit_all = self.raw_nodevalue_func(child_maze, prev_mazes, *self.node_params) 
                    value = parent_to_child + child_to_exit_all
                    raw_values.append(value)
                
                #Apply whatever function you want to these raw values
                probs = self.parent_nodeprob_func(raw_values, *self.parent_params) 
                probs_summary[parent_node] = {child_node: prob for child_node,prob in zip(children, probs)}
            result.append(probs_summary)
        return result
    




class DecisionModelRange(DecisionModel):
    """A model class, with a range of possible parameters."""
    
    def __init__(self, model_name, 
                 node_params_ranges, parent_params_ranges, evaluation_function,
                 raw_nodevalue_func=blind_nodevalue_comb, parent_nodeprob_func=softmax,):
        
        super().__init__(self, model_name, raw_nodevalue_func=blind_nodevalue_comb, parent_nodeprob_func=softmax)
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
        
        #Generate all of the models we want
        models = []
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
            

if __name__ == '__main__':
    Maze1 = mazes['1']
    node_params = (1, 1, .5, .5)
    model = DecisionModel("memory", node_params=node_params, raw_nodevalue_func=blind_nodevalue_with_memory)
    vals = model.choice_probs([Maze1, Maze1])
    print(vals)