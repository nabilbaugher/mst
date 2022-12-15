# Random data analysis of our trials_rows.csv
# Using data_parser.py

from data_parser import *
import csv


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
    
subject_decisions = decisions_to_subject_decisions(decisions)

print(decisions.keys())

m = decisions['0e9aebe0-7972-11ed-9421-5535258a0716']
    
n = m['1']

ex_decisions = subject_decisions['0e9aebe0-7972-11ed-9421-5535258a0716']

Maze1 = mazes['1']

maze = Maze1

for s in n['path']:
    maze = maze.update_map(pos=s)
    print(s)
    if s in n['node_changes']:
        print("Bazinga uwu") #New node!
    
    maze.visualize(s)
