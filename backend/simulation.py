import matplotlib.pyplot as plt
import pprint
import numpy as np
import matplotlib.colors as colors

from maze import Maze, maze2graph, grid2maze
from decisionmodel import DecisionModel, blind_nodevalue_with_memory, blind_nodevalue_comb
from test_mazes import mazes, graphs

"""
The goal of this file is to run maze simulations based on our models, and to visualize the results.
========================================================

Functions:
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

def best_paths(mazes, model):
    """
    Get the best path for each of the mazes in order based on the model
    """
    result = []
    probs_summary = model.choice_probs(mazes)
    for i, maze in enumerate(mazes):
        path = [maze]
        probs_for_maze = probs_summary[i]
        
        while True:
            cur = path[-1]
            if not cur.exit: # exit can be seen
                break
            next_node = max(probs_for_maze[cur], key=lambda next_node: probs_for_maze[cur][next_node])
            path.append(next_node)
        result.append([node.pos for node in path])
    # if len(mazes) == 2:
    #     print(result[1])
    return result


def visualize_juxtaposed_best_paths(mazes, models, index):
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
    current_maze = mazes[str(index+1)]
    prev_mazes = [mazes[str(i+1)] for i in range(index)]
    
    if type(current_maze)==tuple: # Convert to desired format
        current_maze=Maze(current_maze)
        
    # Empty plot to draw on
    fig, axs = plt.subplots(1, len(models))
    fig.suptitle('Maze ' + str(index+1))
    axs = axs.flat
    
    mazes = prev_mazes + [current_maze]

    # Each value function gets draw on its own map
    for ax, model in zip(axs, models):
        paths = best_paths(mazes, model) # Get the best path
        path_for_current_maze = paths[index]
        current_maze.visualize(ax=ax, path=path_for_current_maze) # Draw the maze
        model_name = model.model_name # Label the maze
        ax.set_title(model_name)
    plt.show()
    

if __name__ == "__main__":
    # Test the visualization
    # 0.0, 1.5, 0.5, 0.1
    memory_model = DecisionModel("memory", node_params=(0, 1.5, .5, .1), parent_params=(10,), raw_nodevalue_func=blind_nodevalue_with_memory)
    forget_model = DecisionModel("non memory", node_params=(1, 1), parent_params=(10,), raw_nodevalue_func=blind_nodevalue_comb)
    for i in range(30):
        visualize_juxtaposed_best_paths(mazes, [memory_model, forget_model], i)