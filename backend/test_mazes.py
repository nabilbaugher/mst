from maze import Maze, grid2maze

"""
This file contains several of our mazes we can test our code with, making sure it runs correctly.
"""

map_1 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
         (3, 0, 0, 0, 3, 2, 0, 0, 3),
         (3, 3, 0, 3, 3, 3, 0, 3, 3),
         (3, 6, 6, 6, 5, 6, 6, 6, 3),
         (3, 3, 0, 3, 3, 3, 0, 3, 3),
         (3, 0, 0, 0, 3, 0, 0, 0, 3),
         (3, 3, 3, 3, 3, 3, 3, 3, 3))
         
        

map_2 = ((3, 3, 3, 3, 3, 3, 3),
         (3, 0, 0, 6, 0, 0, 3),
         (3, 2, 3, 6, 3, 0, 3),
         (3, 3, 3, 5, 3, 3, 3),
         (3, 0, 3, 6, 3, 0, 3),
         (3, 0, 0, 6, 0, 0, 3),
         (3, 3, 3, 3, 3, 3, 3))

map_3 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
         (3, 0, 0, 3, 6, 0, 0, 0, 3),
         (3, 0, 3, 3, 6, 3, 3, 0, 3),
         (3, 0, 3, 3, 6, 3, 3, 3, 3),
         (3, 6, 6, 6, 5, 6, 6, 6, 3),
         (3, 3, 3, 3, 6, 3, 3, 0, 3),
         (3, 2, 3, 3, 6, 3, 3, 0, 3),
         (3, 0, 0, 0, 6, 3, 0, 0, 3),
         (3, 3, 3, 3, 3, 3, 3, 3, 3))

map_4 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
         (3, 0, 0, 0, 3, 0, 0, 0, 3),
         (3, 0, 3, 0, 3, 0, 3, 0, 3),
         (3, 6, 5, 6, 6, 6, 6, 6, 3),
         (3, 0, 3, 0, 3, 0, 3, 0, 3),
         (3, 0, 0, 0, 3, 0, 2, 0, 3),
         (3, 3, 3, 3, 3, 3, 3, 3, 3))

map_5 = ((3, 3, 3, 3, 3, 3, 3),
         (3, 3, 3, 5, 3, 3, 3),
         (3, 0, 3, 6, 3, 0, 3),
         (3, 0, 0, 6, 0, 0, 3),
         (3, 3, 3, 6, 3, 3, 3),
         (3, 0, 3, 6, 3, 2, 3),
         (3, 0, 0, 6, 0, 0, 3),
         (3, 3, 3, 3, 3, 3, 3))
##Maze 5 has two root nodes at the beginning
#Maze 6 has a disjoint path

map_6 = ((3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
         (3, 3, 3, 3, 0, 0, 3, 3, 0, 0, 3),
         (3, 3, 3, 3, 0, 0, 3, 3, 0, 0, 3),
         (3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3),
         (3, 6, 6, 6, 6, 6, 6, 6, 6, 5, 3),
         (3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3),
         (3, 2, 0, 3, 3, 0, 0, 3, 3, 3, 3),
         (3, 0, 0, 3, 3, 0, 0, 3, 3, 3, 3),
         (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3))

map_7 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
         (3, 3, 3, 3, 0, 3, 0, 3, 3),
         (3, 3, 3, 3, 0, 3, 0, 3, 3),
         (3, 5, 6, 6, 6, 6, 6, 6, 3),
         (3, 6, 3, 3, 3, 3, 3, 6, 3),
         (3, 6, 6, 6, 6, 6, 6, 6, 3),
         (3, 3, 0, 0, 3, 3, 3, 3, 3),
         (3, 3, 3, 3, 3, 3, 3, 3, 3),)



map_8 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
         (3, 3, 3, 3, 0, 2, 3, 3, 3),
         (3, 0, 3, 0, 0, 0, 3, 0, 3),
         (3, 6, 6, 6, 5, 6, 6, 6, 3),
         (3, 0, 3, 0, 0, 0, 3, 0, 3),
         (3, 3, 3, 3, 0, 3, 3, 3, 3),
         (3, 3, 3, 3, 3, 3, 3, 3, 3))


map_9 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
         (3, 0, 0, 0, 3, 0, 0, 0, 3),
         (3, 0, 3, 0, 3, 0, 3, 0, 3),
         (3, 6, 5, 6, 6, 6, 6, 6, 3),
         (3, 0, 3, 0, 3, 0, 3, 0, 3),
         (3, 0, 0, 0, 3, 0, 2, 0, 3),
         (3, 3, 3, 3, 3, 3, 3, 3, 3))

Maze1, Maze2 = grid2maze(map_1), grid2maze(map_2)
    