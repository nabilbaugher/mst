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
    
    TREE = map2tree(map_) 
    value_summary = node_values(map_, parameters, raw_nodevalue_func)
    
    nid = TREE[0]
    node = 0
    
    path = [node] # list of node ids

    while True:

        # break if at leaf node
        print(nid)
        if not TREE[node]['children']:
            return path

        # non-decision (num(children) <= 1) nodes just gets added
        if len(TREE[node]['children']) == 1:
            path.append(node)
            print(TREE[node])
            node = list(TREE[node]['children'])[0]
            continue

        # find best child
        #print(value_summary)
        best_child = max([ (value, cid) for cid,value in value_summary[node][1].items() ])[1]
        # update node
        node = best_child

        # update path
        path.append(node)


def map_visualizer(map_, node=None):
    """
    Turns a map representation into a human-interpretable image.

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
            -tuple of tuples    has length = nrows, 
            -each tuple of ints has length = ncols.
           
    node : int, optional
        Visualize a specific node for this map: in other words, show a partially explored map. 
        node is simply the number id for one of these partial paths.
            -Note: If the id is too high, there may be no corresponding node.

    Returns
    -------
    None
    
    
    Uses matplotlib.pyplot to make our map human-viewable.
    
    If node is given, the corresponding partially explored map will be displayed. Meaning, in this map,
    our player will have moved around to explore some of the black tiles.
    
    The map structure will still match map_, but it will display the path taken with a dotted line,
    And any squares that have been viewed by this player is 


    """

    nrows, ncols = len(map_), len(map_[0])
    
    curr_map = map_
    
    if node: #If given a node, we've chosen a partial path.
              #Draw that path!
        
        tree = map2tree(map_) #Get tree to get nodes
        
        #Make sure that this node is valid!
        try:
            path = tree[node]['path_from_root']
        except:
            raise ValueError(f"The node value {node} is not in range: this node does not exist!")

        #Get the map matching this node
        curr_map = tree[node]['map']
        curr_pos = tree[node]['pos']
        prev_pos = path[-1]
        
        #Update map based on our last step
        curr_map = update_map(curr_map, curr_pos, prev_pos)
        
        #row --> y axis, col --> x axis
        #Thus, to do (x,y), we need tuples of the form (c,r)
        path = [(c ,r) for r ,c in path][::-1]
        #Also reversing order of path
        
        #Convert pairs of elements (x,y) into two lists: X and Y
        X, Y = zip(*[ (x + 0.5, nrows - y - 0.5)   for x ,y in path])
        #Offset (+0.5, -0.5) is so that our dots are centered on each tile
        
        
        ###Plotting our path
        
        #Draw dotted line between each tile on path
        ax.plot(X, Y, 'o--',  markersize=4, label=node)
        
        #Color our starting point (X[-1],Y[-1]) as purple
        ax.plot(X[-1], Y[-1], 's', markersize=8, color='purple')




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

"ax already has the drawing of the maze, path is in terms of node ids"
def visualize_path(map_, path, ax):
    """
    Takes a visual map, and draws a path on that map from start to end.

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
            -tuple of tuples    has length = nrows, 
            -each tuple of ints has length = ncols.
            
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

    # draw paths
    TREE=map2tree(map_)
    #print(TREE)
    for nid in path[1:]:
        c, r = zip(*[(c + 0.5, nrows - r - 0.5) for r,c in TREE[nid]['path_from_par']])
        c, r = jitter(c), jitter(r)
        ax.plot(c, r, 'o--',  markersize=4, label=nid)
        # ax.plot(x[0], y[0], 's', markersize=8, color='purple')

    ax.legend(loc='upper left', bbox_to_anchor=(0,-0.1))


def visualize_juxtaposed_best_paths(maze):

    _, axs = plt.subplots(1, 3)
    axs = axs.flat

    raw_nodevalue_func_and_params = [('Expected_Utility', raw_nodevalue_comb, (1,1,1)),
                                     ('Discounted_Utillity', raw_nodevalue_comb, (1,1,1)),
                                     ('Probability_Weighted', raw_nodevalue_comb, (1,1,1))]

    for ax, (model_name, raw_nodevalue_func, params) in zip(axs, raw_nodevalue_func_and_params):

        path = best_path(maze, params, raw_nodevalue_func )
        visualize_maze(maze, ax)
        visualize_path(maze, path, ax)
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