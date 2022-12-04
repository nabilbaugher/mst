import matplotlib.pyplot as plt
import pprint
import numpy as np
import matplotlib.colors as colors
import pickle
import time
import itertools

from mst_prototype import map2tree, node_values, map_builder

from mst_prototype import raw_nodevalue_comb

pp = pprint.PrettyPrinter(compact=False)

import models

#with open(f'__experiment_1/parsed_data/tree.pickle', 'rb') as handle:
#     TREE = pickle.load(handle)

"returns the full trajectory of best path based on some node value function and parameter"

def best_path(map_, parameters, raw_nodevalue_func):
    """
    Finds the best path, based on our nodevalue function and parameters.

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
            -tuple of tuples    has length = nrows, 
            -each tuple of ints has length = ncols.
            
            Our "maze" the player is moving through.
            
    parameters : tuple of three floats.
                 parameters to help calculate values. 
                 
                 If using raw_nodevalue_func=raw_nodevalue_comb, 
                 Contains our parameters (tau, gamma, beta).
            
    raw_nodevalue_func : function, optional
        A function that computes the value of a single node.
        Configurable, so we can try out different functions/parameters for computing node values.
        
    Returns
    -------
    path : list of ints
        List of nodes, from start to end of the path, referenced by their node id.
        
        This path is the best path.

    """
    
    TREE = map2tree(map_)  #Generate tree
    value_summary = node_values(map_, parameters, raw_nodevalue_func) #Get all of our values
    
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

def visualize_maze(map_, ax=None):
    """
    Turns a map representation into a human-interpretable image. 
    Modifies the input ax object, which is what we draw the map onto.

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
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


    nrows, ncols = len(map_), len(map_[0])


    #Rows count going down: we have to reverse our map order.
    maze1=map_[::-1]
    
    pp.pprint(maze1)

    #Gather color map
    cmap = colors.ListedColormap(['#9c9c9c', 'white', '#d074a4', '#b0943d', 'white', '#a1c38c', 'white', '#f5f5dc'])
    boundaries = [0, 1, 2, 3, 4, 5, 6, 7]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=False)

    #Draw maze using a mesh
    ax.pcolormesh(maze1, edgecolors='lightgrey', linewidth=1, cmap=cmap, norm=norm)
    ax.set_aspect('equal')


    # Major ticks positions
    #+.5 so they're centered on each square
    ax.set_xticks([x+0.5 for x in range(ncols)])
    ax.set_yticks([y+0.5 for y in range(nrows)])
    
    
    # Major ticks label (for readability of plot, (0,0) at top left)
    ax.set_xticklabels([y for y in range(ncols)])
    ax.set_yticklabels([y for y in range(nrows)][::-1]) #Reverse order because rows count down, not up


# with open(f'__experiment_1/mazes/{world}.txt') as f:
#     ncols = int(f.readline())
#     nrows = int(f.readline())


def visualize_path(map_, path, ax):
    """
    Takes a visual map, and draws a path on that map from start to end. 
    The visual, represented as ax, is modified in-place: it doesn't have to be returned.

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
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
    nrows, ncols = len(map_), len(map_[0])

    def jitter(arr):
        """Makes our lines look a little wobbly."""
        stdev = .025 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    #Get a tree
    TREE=map2tree(map_)
    
    
    for node in path[1:]: #Go through each node
    
        parent_to_child_path = TREE[node]['path_from_par']
    
        c, r = zip(*[ (c + 0.5, nrows - r - 0.5)  #Offset by 0.5 centers our path lines on each grid square
                     for r,c in parent_to_child_path]) #Each step from the parent to child
        
        c, r = jitter(c), jitter(r) #Make each step a little wobbly for visuals
        
        ax.plot(c, r, 'o--',  markersize=4, label=node) #Draw the path from parent to child

    ax.legend(loc='upper left', bbox_to_anchor=(0,-0.1)) #Add legend
    

###(tau, gamma, beta)

eu_du_pwu = [('Expected_Utility', raw_nodevalue_comb, (1,1,1)),
             ('Discounted_Utillity', raw_nodevalue_comb, (1,1,1)),
             ('Probability_Weighted', raw_nodevalue_comb, (1,1,1))]

def visualize_juxtaposed_best_paths(map_, raw_nodevalue_func_and_params= eu_du_pwu):
    """
    This function allows us to visually compare the "best paths" chosen by multiple different
    value functions.

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
            -tuple of tuples    has length = nrows, 
            -each tuple of ints has length = ncols.
            
            Our "maze" the player is moving through.
        
    raw_nodevalue_func_and_param: list of tuples ( str, function, tuple(float, float float) )
    
            -list of tuples: each tuple represents a different way to measure the best value 
                -str: The name of the value function approach
                
                -function: The value function we use for each node.
                
                -tuple(float,float,float): The parameters for our value function.
                    -Assuming we use raw_nodevalue_comb, these parameters are (tau, gamma, beta)
            
            Contains all of the different value functions we want to compare.

    Returns
    -------
    None.

    """
    num_funcs = len(raw_nodevalue_func_and_params)
    #Empty plot to draw on
    _, axs = plt.subplots(1, num_funcs)
    axs = axs.flat

    #Each value function gets draw on its own map
    for ax, (model_name, raw_nodevalue_func, params) in zip(axs, raw_nodevalue_func_and_params):
        
        #Get the best path
        path = best_path(map_, params, raw_nodevalue_func )
        #Draw our maze
        visualize_maze(map_, ax)
        #Draw the path on our maze
        visualize_path(map_, path, ax)
        #Label this maze
        ax.set_title(model_name)

    plt.show()



if __name__ == "__main__":

    world = '4ways'
    world= ((3, 3, 3, 3, 3, 3, 3, 3, 3),
         (3, 3, 3, 3, 0, 3, 0, 3, 3),
         (3, 3, 3, 3, 0, 3, 0, 3, 3),
         (3, 5, 6, 6, 6, 6, 6, 6, 3),
         (3, 6, 3, 3, 3, 3, 3, 6, 3),
         (3, 6, 6, 6, 6, 6, 6, 6, 3),
         (3, 3, 0, 0, 3, 3, 3, 3, 3),
         (3, 3, 3, 3, 3, 3, 3, 3, 3),)
    #
    # world= ((3, 3, 3, 3, 3, 3, 3, 3, 3),
    #      (3, 3, 3, 3, 0, 3, 0, 3, 3),
    #      (3, 3, 3, 3, 0, 3, 0, 3, 3),
    #      (3, 6, 5, 6, 6, 6, 6, 6, 3),
    #      (3, 6, 3, 3, 3, 3, 3, 6, 3),
    #      (3, 6, 6, 6, 6, 6, 6, 6, 3),
    #      (3, 3, 0, 0, 3, 3, 3, 3, 3),
    #      (3, 3, 3, 3, 3, 3, 3, 3, 3),)


    comp1 = ((3,3,3,3,3,3,3,3,3),
             (3,3,3,3,0,2,3,3,3),
             (3,0,3,0,0,0,3,0,3),
            (3,6,6,6,5,6,6,6,3),
            (3,0,3,0,0,0,3,0,3),
            (3,3,3,3,0,3,3,3,3),
             (3,3,3,3,3,3,3,3,3),)

    comp2=((3,3,3,3,3,3,3,3,3),
        (3,0,0,0,3,0,0,0,3),
           (3,0,3,0,3,0,3,0,3),
           (3,6,5,6,6,6,6,6,3),
           (3,0,3,0,3,0,3,0,3),
           (3,0,0,0,3,0,2,0,3),
           (3,3,3,3,3,3,3,3,3),)

    Maze1 = ((3,3,3,3,3,3,3,3,3),
            (3,0, 0, 0, 3, 2, 0, 0,3),
            (3,3, 0, 3, 3, 3, 0, 3,3),
            (3,6, 6, 6, 5, 6, 6, 6,3),
            (3,3, 0, 3, 3, 3, 0, 3,3),
            (3,0, 0, 0, 3, 0, 0, 0,3),
            (3,3,3,3,3,3,3,3,3),)

    Maze2=((3,3,3,3,3,3,3),
           (3,0, 0, 6, 0, 0,3),
            (3,0, 3, 6, 3, 2,3),
           (3,3, 3, 5, 3, 3,3),
            (3,0, 3, 6, 3, 0,3),
            (3,0, 0, 6, 0, 0,3),
           (3,3,3,3,3,3,3),)

    Maze3=((3,3,3,3,3,3,3,3,3),
           (3,0, 0, 3, 6, 0, 0, 0,3),
           (3,0, 3, 3, 6, 3, 3, 0,3),
           (3,0, 3, 3, 6, 3, 3, 3,3),
           (3,6, 6, 6, 5, 6, 6, 6,3),
           (3,3, 3, 3, 6, 3, 3, 0,3),
           (3,2, 3, 3, 6, 3, 3, 0,3),
           (3,0, 0, 0, 6, 3, 0, 0,3),
           (3,3,3,3,3,3,3,3,3),)

    Maze4=((3,3,3,3,3,3,3,3,3),
           (3,0, 0, 0, 3, 0, 0, 0, 3),
           (3,0, 3, 0, 3, 0, 3, 0, 3),
           (3,6, 5, 6, 6, 6, 6, 6, 3),
           (3,0, 3, 0, 3, 0, 3, 0, 3),
           (3,0, 0, 0, 3, 0, 2, 0, 3),
           (3,3,3,3,3,3,3,3,3),)

    Maze5=((3,3,3,3,3,3,3),
           (3,3, 3, 5, 3, 3, 3),
           (3,0, 3, 6, 3, 0, 3),
           (3,0, 0, 6, 0, 0, 3),
           (3,3, 3, 6, 3, 3, 3),
           (3,0, 3, 6, 3, 2, 3),
           (3,0, 0, 6, 0, 0, 3),
           (3,3,3,3,3,3,3),)
    ##Maze 5 has two root nodes at the beginning
    #Maze 6 has a disjoint path

    Maze6=((3,3,3,3,3,3,3,3,3,3,3),
           (3,3, 3, 3, 0, 0, 3, 3, 0, 0, 3),
           (3,3, 3, 3, 0, 0, 3, 3, 0, 0, 3),
           (3,3, 3, 3, 0, 3, 3, 3, 0, 3, 3),
           (3,6, 6, 6, 6, 6, 6, 6, 6, 5, 3),
           (3,3, 0, 3, 3, 3, 0, 3, 3, 3, 3),
           (3,2, 0, 3, 3, 0, 0, 3, 3, 3, 3),
           (3,0, 0, 3, 3, 0, 0, 3, 3, 3, 3),
           (3,3,3,3,3,3,3,3,3,3,3),)

    visualize_juxtaposed_best_paths(Maze1)