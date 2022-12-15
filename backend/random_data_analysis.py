# Random data analysis of our trials_rows.csv
# Using data_parser.py

from data_parser import *
import csv
import pandas as pd


# Things that we want to learn:

# Average total number of steps for each maze for humans
# Average total number of steps for each maze for models

# Average human time, standard deviation

# Mazes changing over time
#   Find some way to represent the fact that the first 15 mazes have same exit locations
#   Second one has different exit locations
#   Graph, for one of our best models, human performance on each maze, maybe up then down

# How correlated is memoryless performance vs memory performance

# Previous models
#   Make the step cells equations
#   Make sure the other previous equations work on our mazes

# Run the previous models on the new mazes:
# * Expected utility
# * Discounted utility
# * Probability weighted utility
# * Combined
# * Heuristics

# Cool things to do:
# Track backtracking - did they do "left" then "right"
# See how similar human traversals are to each other



#
# Starting analysis

# Average total number of steps for each maze for humans
# Average total number of steps for each maze for models

# Average human time, standard deviation

trial_data = get_csv("./data/prolific_data_sorted_tester_id_created_at")

decisions = convert_data(trial_data)
    
# subject_decisions = decisions_to_subject_decisions(decisions)

tester_id = '0e9aebe0-7972-11ed-9421-5535258a0716'


print("decisions")
print(decisions)

print("tester_id")
print(decisions[tester_id]["1"])

# Get lengths of each path

# def get_path_length():
#     # Given a dictionary returned by convert_data(trial_data)[tester_id][maze_number]
#     # Get the path length
#     pass

def filter_data_redundant_trials(csv_name):
    """
    Given a csv filepath, remove redundant trials (sometimes, there are multiple trials created that differ
    simply in the number of "won" statements at the end)

    Exports a csv with redundant trials removed
    """
    trial_data_filtered = pd.read_csv(csv_name)

    current_trial_number = '-1'
    for index, row in trial_data_filtered.iterrows():
        keystrokes = eval(row["keystroke_sequence"])
        this_row_trial = list(keystrokes.keys())[0]
        if this_row_trial == current_trial_number:
            trial_data_filtered = trial_data_filtered.drop(index)
        else:
            # Move to next keystroke sequence key
            current_trial_number = this_row_trial

    trial_data_filtered.to_csv('./data/prolific_data_filtered.csv')


filter_data_redundant_trials('./data/prolific_data_sorted_tester_id_created_at.csv')



def get_path_lengths_for_all_users():
    """ Given a dictionary given by convert_data(trial_data)
    Return the path length of all users for all mazes

    Input:
        subject_decisions: dictionary where key is subject, value is a decisions_list
        decisions_list : list of tuples (Maze, Maze)
            Each tuple represents one decision our player made, moving between two nodes: (parent_maze, child_maze)
            To reach this node, our player has to have gone from its parent node, and chosen this option.
            
            This list in total represents every decision our player made.

    Output:
        Dictionary

        Key is tester_id, value is an array with all of the 30 path lengths
    """
    


# Visualizing the human traversal through the map

# print(decisions.keys())

# tester_id = '0e9aebe0-7972-11ed-9421-5535258a0716'

# m = decisions[tester_id]
    
# n = m['1']

# ex_decisions = subject_decisions[tester_id]

# Maze1 = mazes['1']

# maze = Maze1

# for s in n['path']:
#     maze = maze.update_map(pos=s)
#     print(s)
#     if s in n['node_changes']:
#         print("Bazinga uwu") #New node!
    
#     maze.visualize(pos=s)