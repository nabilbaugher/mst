import matplotlib.pyplot as plt
import pprint
import numpy as np
import matplotlib.colors as colors

from maze import Maze, maze2graph, grid2maze
from decisionmodel import DecisionModel
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
    for maze in mazes:
        graph = graphs[maze.name]
        probs_summary = model.choice_probs(graph)
        path = [maze.start]
        
        while True:
            cur = path[-1]
            
            option = probs_sumam
            
            if cur == maze.end:
                break
            next_node = max(probs_summary[cur], key=lambda next_node: probs_summary[cur][next_node])
            path.append(next_node)
        result.append(path)
    return result


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
    if type(maze)==tuple: # Convert to desired format
        maze=Maze(maze)
        
    num_funcs = len(models)
    # Empty plot to draw on
    _, axs = plt.subplots(1, num_funcs)
    axs = axs.flat

    # Each value function gets draw on its own map
    for ax, model in zip(axs, models):
        path = best_paths(maze, model) # Get the best path
        visualize_maze(maze, ax) # Draw the maze
        visualize_path(maze, path, ax) # Draw the path on the maze
        model_name = model.model_name # Label the maze
        ax.set_title(model_name)

    plt.show()