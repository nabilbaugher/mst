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
from avg_log_likelihood import avg_log_likelihood_decisions

# Circular import so added copy of function to this file
# from model_evaluation import avg_log_likelihood_decisions 

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

def softmax_complement(values, tau):
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
    # numer = [np.exp(v * ( 1 /tau)) if v != float('inf') else 0 for v in values]
    numer = [np.exp(v * ( 1 /tau)) for v in values]
    denom = sum(numer)
    if len(numer) == 1:
        return [1]
    for n in numer:
        if n == 0:
            raise ValueError("numerator is 0")
    for n in numer:
        if n == sum(numer):
            print(len(numer), n, sum(numer))
            print(values)
            print(tau)
            raise ValueError("numerator is num of numerator")

    numer = [ 1 - n/denom for n in numer]
    denom = sum(numer)
    return [n/denom for n in numer]



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


def random_heuristic(maze, prev_mazes=None, oops=None):
    return np.random.random()

def steps_cells_heuristic(maze, prev_mazes=None, step_weight=1):
    """
    Loss = step_weight * number of steps taken - number of cells revealed
    """
    graph = graphs[maze.name]
    
    parent_node = maze
    parent_hidden = maze.get_hidden()
    
    child_losses = {} # Values for each child we could choose
    children = graph[parent_node]["children"] # All children

    for child_maze, path_to_child in children.items():
        child_hidden = child_maze.get_hidden()
        num_cells_revealed = len(parent_hidden) - len(child_hidden)
        num_steps_taken = len(path_to_child)
        child_losses[child_maze] = step_weight * num_steps_taken - num_cells_revealed
    
    if children == {}:
        return 0
    
    true_loss = min(child_losses.values())
    return true_loss


def steps_cells_heuristic_with_memory(maze, prev_mazes, step_weight=1, memory_weight=0.5, learning_rate=0.5):
    """
    Loss = memory_weight * Q_memory + (1-memory_weight) * steps_cells_heuristic
    """
    Q_steps_cells = steps_cells_heuristic(maze, prev_mazes, step_weight)
    if len(prev_mazes) == 0:
        return Q_steps_cells

    Q_past_all = weighted_distance_to_previous_exits(maze, prev_mazes, learning_rate)
    # if child_maze.pos == (8,11) and len(prev_mazes) == 4:
    #     visualize_2d_array(dictionary_to_matrix(Q_past_all, child_maze.nrows, child_maze.ncols))
    Q_past = Q_past_all[maze.pos]
    
    if memory_weight == 0 and Q_past == float('inf'):
        return (1 - memory_weight) * Q_steps_cells
    
    if Q_past == float('inf'):
        return Q_steps_cells
    
    Q_mem = memory_weight * Q_past + (1 - memory_weight) * Q_steps_cells
    if Q_mem == float('inf'):
        raise Exception("Q_mem is inf")
    return Q_mem


@memoize #Hopefully saves on time cost to compute?
def blind_nodevalue_comb(maze, prev_mazes=None, gamma=1, beta=1):
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
    
    parent_node = maze
    parent_hidden = maze.get_hidden()
    
    child_losses = {} # Values for each child we could choose
    children = graph[parent_node]["children"] # All children

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
        
        necessary_loss = parent_to_child # No matter what, we will get this loss: we have to walk to the child to find out whether we win.
        success_loss = weighted_prob * child_to_exit # Loss assuming we find the exit
        failure_loss = gamma * weighted_comp *  child_to_exit_later # Loss assuming we don't find the exit at this child node
        
        #Expected loss: each loss has been scaled by its probability of occurring
        child_loss = necessary_loss + success_loss + failure_loss
        child_losses[child_maze] = child_loss #All losses
    
    if children == {}:
        return 0
    
    true_loss = min(child_losses.values()) # We pick the child that incurs the least loss
    if true_loss == float('inf'):
        raise ValueError("Infinite loss")
    return true_loss


def get_distances_to_exit_in_optimal_path(maze, exit_):
    """
    Get the distance from each node in the optimal path to the exit.
    """
    nrows = maze.nrows
    ncols = maze.ncols
    result = [[float('inf') for _ in range(ncols)] for _ in range(nrows)]
    result[exit_[0]][exit_[1]] = 0
    non_walls = set(maze.black) | set(maze.path) | set([maze.exit]) | set([maze.start])
    
    # bfs
    q = deque() # (row, col, parent_row, parent_col, distance)
    q.append((exit_[0], exit_[1], None, None, 0))
    parents = {}
    while q:
        row, col, parent_row, parent_col, distance = q.popleft()
        if (row, col) in parents:
            continue
        if (row, col) not in non_walls:
            continue

        parents[(row, col)] = (parent_row, parent_col)
        if (row, col) == maze.start:
            break
        q.append((row + 1, col, row, col, distance + 1))
        q.append((row - 1, col, row, col, distance + 1))
        q.append((row, col + 1, row, col, distance + 1))
        q.append((row, col - 1, row, col, distance + 1))
    
    row, col = maze.start
    while (row, col) != exit_:
        row, col = parents[(row, col)]
        result[row][col] = distance
        distance -= 1

    return result


@memoize
def distance_to_exit(maze, point):
    """
    Get the distance from a point to the exit using bfs.
    """
    non_walls = set(maze.black) | set(maze.path) | set([maze.exit]) | set([maze.start])
    q = deque() # (row, col, distance)
    q.append((point[0], point[1], 0))
    while q:
        row, col, distance = q.popleft()
        if (row, col) == maze.exit:
            return distance
        if (row, col) not in non_walls:
            continue
        non_walls.remove((row, col))
        q.append((row + 1, col, distance + 1))
        q.append((row - 1, col, distance + 1))
        q.append((row, col + 1, distance + 1))
        q.append((row, col - 1, distance + 1))
    return float('inf')    


def weighted_distance_to_previous_exits(current_maze, prev_mazes, learning_rate=.5, weights=None):
    """
    For each previous maze, calculate the value of each position on the current maze's path
    based on the position of the exit in the previous maze. The value is the distance to the exit.
    If the exit is not in the current maze, the value is the distance to the closest point to the exit
    plus the distance from that point to the exit.
    """
    if not weights:
        weights = {}
    
    # Base case: If there are no more previous mazes, return the weights
    if not prev_mazes:
        return weights
    
    non_walls = set(current_maze.black) | set(current_maze.path) | set([current_maze.start])
    if current_maze.exit is not None:
        non_walls |= set([current_maze.exit])
    
    prev_maze = prev_mazes[0]
    # if the exit can be seen in the current maze, return 0
    # if current_maze.exit is None:
    #     return 0
    
    if prev_maze.exit in non_walls:
        # If the exit is in the current maze, the value is the distance to the exit
        distances = get_distances_to_exit_in_optimal_path(current_maze, prev_maze.exit)
        for p in non_walls:
            if p in weights:
                weights[p] += distances[p[0]][p[1]]
            else:
                weights[p] = distances[p[0]][p[1]]
    else:
        # If the exit is not in the current maze, the value is the distance to the closest point
        # to the exit plus the distance from that point to the exit
        closest_point = min(non_walls, key=lambda p: distance_to_exit(prev_maze, p))
        distances = get_distances_to_exit_in_optimal_path(current_maze, closest_point)
        distance_from_closest_point_to_exit = abs(closest_point[0] - prev_maze.exit[0]) + abs(closest_point[1] - prev_maze.exit[1])
        for p in non_walls:
            if p in weights:
                weights[p] += distances[p[0]][p[1]] + distance_from_closest_point_to_exit
            else:
                weights[p] = distances[p[0]][p[1]] + distance_from_closest_point_to_exit

    # Update the weights using the learning rate
    updated_weights = {}
    for p in weights:
        updated_weights[p] = weights[p] * learning_rate
    weights = updated_weights
    
    # Recursively call the function on the remaining previous mazes
    return weighted_distance_to_previous_exits(current_maze, prev_mazes[1:], learning_rate, weights)


def dictionary_to_matrix(dictionary, nrows, ncols):
    """
    Convert a dictionary of (row, col) to value to a matrix.
    """
    result = [[float('inf') for _ in range(ncols)] for _ in range(nrows)]
    for (row, col), value in dictionary.items():
        result[row][col] = value
    return result


@memoize #Hopefully saves on time cost to compute?
def blind_nodevalue_with_memory(child_maze, prev_mazes, discount_factor=1, bad_estimation_factor=1, memory_weight=.5, learning_rate=.5):
    """
    Q_blind = blind nodevalue no memory
    Q_past = nodevalue based on past maps exits, and current map layout
    Q_mem = weighted average of Q_seen and Q_past
    """
    Q_blind = blind_nodevalue_comb(child_maze, discount_factor, bad_estimation_factor) # positive slope

    if len(prev_mazes) == 0:
        return Q_blind

    Q_past_all = weighted_distance_to_previous_exits(child_maze, prev_mazes, learning_rate)
    # if child_maze.pos == (8,11) and len(prev_mazes) == 4:
    #     visualize_2d_array(dictionary_to_matrix(Q_past_all, child_maze.nrows, child_maze.ncols))
    Q_past = Q_past_all[child_maze.pos]
    
    if memory_weight == 0 and Q_past == float('inf'):
        return (1 - memory_weight) * Q_blind
    
    if Q_past == float('inf'):
        return Q_blind
    
    Q_mem = memory_weight * Q_past + (1 - memory_weight) * Q_blind
    if Q_mem == float('inf'):
        raise Exception("Q_mem is inf")
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
                 raw_nodevalue_func=blind_nodevalue_comb, parent_nodeprob_func=softmax_complement):
        
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

        Returns list of dicts of dicts
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
                    if value > 1000:
                        raise Exception("infinite expected distance?")
                
                #Apply whatever function you want to these raw values
                probs = self.parent_nodeprob_func(raw_values, *self.parent_params) 
                for prob in probs:
                    if prob == 0:
                        raise Exception("We have a zero likelihood!")
                if sum(probs) > 1.01:
                    raise Exception("We have a probability greater than 1!")
                probs_summary[parent_node] = {child_node: prob for child_node,prob in zip(children, probs)}
            result.append(probs_summary)
            # testing
            for pre_choice in probs_summary:
                if maze.map[pre_choice.pos[0]][pre_choice.pos[1]] == 3:
                    raise Exception("We're in a wall!")
                
        return result
    




class DecisionModelRange(DecisionModel):
    """A model class, with a range of possible parameters."""
    
    def __init__(self, model_name, 
                 node_params_ranges, parent_params_ranges, evaluation_function,
                 raw_nodevalue_func=blind_nodevalue_comb, parent_nodeprob_func=softmax_complement,):
        
        super(DecisionModelRange, self).__init__(model_name, raw_nodevalue_func=raw_nodevalue_func, parent_nodeprob_func=parent_nodeprob_func)
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
                
    @memoize
    def fit_parameters(self, decisions_tuple):
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
            print('index:', index+1, 'of', len(models), 'models')
            # print('model params:', model.node_params, model.parent_params)
            evaluation = self.evaluation_function(decisions_tuple, model) #Get model performance
            performance[index] = evaluation #Save each value
            # print('evaluation:', evaluation)
        best_index = max( performance.keys(), key=lambda index : performance[index] ) #Find the best model
        return models[best_index]
            

def visualize_2d_array(arr) -> None:
    plt.show()
    plt.imshow(arr, cmap='seismic')
    plt.colorbar()
    plt.show()
    

if __name__ == '__main__':
    # create DecisionModelRange object with a range of parameters
    node_params_ranges = [(0, 1, 5), (0, 1, 5), (0, 1, 5), (0, 1, 5)]
    decision_model_range = DecisionModelRange("memory", node_params_ranges, None, avg_log_likelihood_decisions)
    
