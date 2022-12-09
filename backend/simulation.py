import matplotlib.pyplot as plt
import pprint
import numpy as np
import matplotlib.colors as colors
import pickle
import time
import itertools

# from mst_prototype import map2tree, node_values, map_builder
# from mst_prototype import raw_nodevalue_comb, softmax
# from loglikes import get_model_and_params
# from map_generator import generate_spiral_maps

from maze import Maze, maze2tree, grid2maze
from decisionmodel import DecisionModel
from test_mazes import mazes, trees

pp = pprint.PrettyPrinter(compact=False)



"""
The goal of this file is to run maze simulations based on our models, and to visualize the results.

========================================================


A short summary of the various functions in this file:
    
    best_path(maze,model): return path
        Based on our maze and model, returns the best path for us to take.
        
        *Uses maze.maze2tree, maze.Maze, decisionmodel.DecisionModel
    
    visualize_maze(maze, ax=None): return None
        Draws a human-readable version of our maze with matplotlib
        
        *Uses maze.Maze
    
    visualize_path(maze, path, ax): return None
        Draws a particular path on top of the matplotlib maze
        
        *Uses maze.maze2tree
        
    visualize_juxtaposed_best_paths(maze, models= eu_du_pwu): return None
        Draw the best path for multiple different models, to compare.
        
        *Uses best_path, visualize_maze, visualize_path
        *Uses maze.Maze, decisionmodel.DecisionModel
        
Reminder of the params typical format
    parent_params_comb = ('tau',)
    node_params_comb = ('gamma','beta')
    
"""


#with open(f'__experiment_1/parsed_data/tree.pickle', 'rb') as handle:
#     TREE = pickle.load(handle)



def best_path(maze, model):
    ####NEEDS TO BE FIXED TO MATCH maze2statetree
    """
    Finds the best path, based on our nodevalue function and parameters.

    Parameters
    ----------
    maze : Maze object, our maze to navigate. Stores a maze.map object:
        
        map: tuple of tuples of ints - represents a grid in the shape (maze.nrows, maze.ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
               Our "maze" the player is moving through.
            
    model : DecisionModel object, representing one model for player-decisions.
    
        Contains several variables:
            str is the name of the model
            
            first tuple has the parameters for node value
            second tuple has the parameters for node probability
            
            first function is used to compute node value
            second function is used to compute node probability
            
        Fully describes a single model for player action.
        
    Returns
    -------
    path : list of ints
        List of nodes, from start to end of the path, referenced by their node id.
        
        This path is the best path.

    """
    
    # model_params = get_model_and_params(model) #Unpack model for node_value purposes
    TREE = maze2tree(maze)  #Generate tree
    
    value_summary = model.node_values(maze) #Get all of our values
    
    node = 0
    
    path = [node] # list of node ids leading down our best past

    while True:

        #If no children, then we're finished! Return this path
        if not TREE[node]['children']:
            return path

        #Get children
        children = TREE[node]['children'] 
        
        #If one child, then we don't have to make a choice: just add the child and move on.
        if len(children) == 1:
            path.append(node) #Add current node
             
            node = list(children)[0] #Grab the only child
            continue 

        #Our actor will take the optimal path
        #So, which child has the largest value?
        
        child_and_value = value_summary[node].items() #Get all children and their values
        
        #Since value is first in tuple, it will be prioritized.
        best_child_and_value = max([ (value, cid) for cid,value in child_and_value ])
        #Remove value, only want the node
        best_child = best_child_and_value[1]
        
        path.append(best_child)
        
        node = best_child





"""
0: hidden, 2: exit, 3: wall, 5: start, 6: open
"""

# with open(f'__experiment_1/mazes/{world}.txt') as f:
#     ncols = int(f.readline())
#     nrows = int(f.readline())
#
#     lines = f.readlines()
#lines=maze

def visualize_maze(maze, ax=None):
    
    """
    Turns a map representation into a human-interpretable image. 
    Modifies the input ax object, which is what we draw the map onto.

    Parameters
    ----------
    maze : Maze object, our maze to navigate. Stores a maze.map object:
        
        map: tuple of tuples of ints - represents a grid in the shape (maze.nrows, maze.ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
               Our "maze" the player is moving through.
            
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The plot on which we will be drawing our map.

    Returns
    -------
    None.

    """
    
    if ax is None:
        _, ax = plt.subplots(1)

    #Rows count going down: we have to reverse our map order.
    maze1=maze.map[::-1]
    
    #pp.pprint(maze1)

    #Gather color map
    cmap = colors.ListedColormap(['#9c9c9c', 'white', '#d074a4', '#b0943d', 'white', '#a1c38c', 'white', '#f5f5dc'])
    boundaries = [0, 1, 2, 3, 4, 5, 6, 7]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=False)

    #Draw maze using a mesh
    ax.pcolormesh(maze1, edgecolors='lightgrey', linewidth=1, cmap=cmap, norm=norm)
    ax.set_aspect('equal')
    

    # Major ticks positions
    #+.5 so they're centered on each square
    ax.set_xticks([x+0.5 for x in range(maze.ncols)])
    ax.set_yticks([y+0.5 for y in range(maze.nrows)])
    
    
    # Major ticks label (for readability of plot, (0,0) at top left)
    ax.set_xticklabels([y for y in range(maze.ncols)])
    ax.set_yticklabels([y for y in range(maze.nrows)][::-1]) #Reverse order because rows count down, not up


# with open(f'__experiment_1/mazes/{world}.txt') as f:
#     ncols = int(f.readline())
#     nrows = int(f.readline())


def visualize_path(maze, path, ax):
    ####NEEDS TO BE FIXED TO MATCH maze2statetree
    """
    Takes a visual map, and draws a path on that map from start to end. 
    The visual, represented as ax, is modified in-place: it doesn't have to be returned.

    Parameters
    ----------
    maze : Maze object, our maze to navigate. Stores a maze.map object:
        
        map: tuple of tuples of ints - represents a grid in the shape (maze.nrows, maze.ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
               Our "maze" the player is moving through.
            
    path : list of ints
        List of nodes, from start to end of the path.
        
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The plot that contains our map. What we will draw our path on.

    Returns
    -------
    None.
    
    """
    #Ignore first position on path
    path=path[1:]

    def jitter(arr):
        """Makes our lines look a little wobbly."""
        stdev = .025 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    #Get a tree
    TREE=maze2tree(maze)
    
    
    for node in path[1:]: #Go through each node
    
        parent_to_child_path = TREE[node]['path_from_par']
    
        c, r = zip(*[ (c + 0.5, maze.nrows - r - 0.5)  #Offset by 0.5 centers our path lines on each grid square
                     for r,c in parent_to_child_path]) #Each step from the parent to child
        
        c, r = jitter(c), jitter(r) #Make each step a little wobbly for visuals
        
        ax.plot(c, r, 'o--',  markersize=4, label=node) #Draw the path from parent to child

    ax.legend(loc='upper left', bbox_to_anchor=(0,-0.1)) #Add legend
    



#Models example

expected_utility_model = DecisionModel(model_name="Expected_Utility")
discounted_utility_model = DecisionModel(model_name="Discounted_Utility")
probability_weighted_model = DecisionModel(model_name="Probability_Weighted")


eu_du_pwu = [expected_utility_model, discounted_utility_model, probability_weighted_model]



def visualize_juxtaposed_best_paths(maze, models= eu_du_pwu):
    """
    This function allows us to visually compare the "best paths" chosen by multiple different
    value functions.

    Parameters
    ----------
    maze : Maze object, our maze to navigate. Stores a maze.map object:
        
        map: tuple of tuples of ints - represents a grid in the shape (maze.nrows, maze.ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
               Our "maze" the player is moving through.
        
    models : list of models
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

    Returns
    -------
    None.

    """
    if type(maze)==tuple: #Convert to desired format
        maze=Maze(maze)
        
    num_funcs = len(models)
    #Empty plot to draw on
    _, axs = plt.subplots(1, num_funcs)
    axs = axs.flat

    #Each value function gets draw on its own map
    for ax, model in zip(axs, models):
        
        #Get the best path
        path = best_path(maze, model )
        #Draw our maze
        visualize_maze(maze, ax)
        #Draw the path on our maze
        visualize_path(maze, path, ax)
        #Label this maze
        model_name = model.model_name
        ax.set_title(model_name)

    plt.show()



if __name__ == "__main__":

    world = '4ways'
    
    
    visualize_juxtaposed_best_paths(grid2maze(mazes['2']))

    #visualize_juxtaposed_best_paths(generate_spiral_maps(1)[0])