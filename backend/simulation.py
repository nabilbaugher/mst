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

def best_path(mazes, model):
    """
    Get the best path for each of the mazes in order based on the model
    """
    best_paths = []
    for maze in mazes:
        graph = graphs[maze.name]
        probs_summary = model.choice_probs(graph)
        path = [maze.start]
        
        while True:
            cur = path[-1]
            
            option = probs_sumam
            
            if cur == maze.end:
                break
            next_node = max( probs_summary, key = lambda cur: probs_summary[])
            
            

        