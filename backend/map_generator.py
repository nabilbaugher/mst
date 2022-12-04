import matplotlib.pyplot as plt
from mst_prototype import map_builder, map_visualizer
from simulation import visualize_maze
import numpy as np
import copy
import random

def generate_spiral_maps(nmaps, nrows, ncols):
    """
    Generates a set of trick maps that all follow the same path to the goal
    but have different extraneous paths.
    
    Args:
        nmaps (int): number of maps to generate
        nrows (int): number of rows in the map
        ncols (int): number of columns in the map
    
    Returns:
        A tuple of tuples (like the return of map_builder).
    """
    # just trying spiral for now
    base_path = generate_spiral_base_path(nrows, ncols)
    trick_maps = []
    for _ in range(nmaps):
        trick_maps.append(map_builder(generate_spiral_map_with_offshoots(nrows, ncols, base_path)))
    return trick_maps


def double_map_size(map):
    """
    Helper function for generate_two_bigger_maps

    Given a map, a tuple of tuples, returns a map that is twice the size by inserting
    a new row every other row and a new column every other column
    """
    double_size_map = [list(x) for x in map] #list of lists

    for row_index in range(len(map) - 1, -1, -1): # Go backward, insert a new row
        # If a row contains 5, the starting location, replace it with a path value 6
        copied_list = [item if item != 5 else 6 for item in list(map[row_index])]
        double_size_map.insert(row_index, copied_list)

    np_array = np.array(double_size_map, dtype=object)
    for col_index in range(len(map[0]) - 1, -1, -1):
        copied_list = [item if item != 5 else 6 for item in list(np_array[:, col_index])]
        
        np_array = np.insert(np_array, col_index, copied_list, axis=1)

    return tuple(tuple(x) for x in np_array)

def generate_different_size_maps(map):
    """
    Given a map with nrows rows and ncolumns columns, create 2 new maps that are larger

    To keep it simple, we are only generating 
        * 1 map that is ~double the size of the original map
        * 1 map that is ~4 times the size of the original map

    We are making a bigger map by copying every column and row
        (If a row or column contains a 5, we replace that 5 with a path tile, 6

    Args:
        map (tuple of tuples): A grid where the number in each space represents:
        * 5 is the start tile
        * 6 is the path tile
        * 0 is the black tile
        * 3 is the wall tile

    Returns:
        A list of maps where each map is a tuple of tuples
        [original_size_map, twice_as_big_map, four_times_as_big_map]
    """

    # Make a copy of the entire map and insert every other row or every other column

    twice_as_big_map = double_map_size(map)
    four_times_as_big_map = double_map_size(copy.deepcopy(twice_as_big_map))

    return [map, twice_as_big_map, four_times_as_big_map]


def generate_spiral_base_path(nrows, ncols, buffer=5):
    """
    Generates a spiral path that all trick maps will follow. 
    
    Args:
        nrows (int): number of rows in the map
        ncols (int): number of columns in the map
        
    Returns:
        A tuple containing
            black (list): a list of tuples (positions of black squares)
            path (list): a list of tuples (positions of path squares)
            start (tuple): the start position
        is_done function for later use
    """
    def in_bounds(row, col):
        # buffer of one tile around the edge
        return 1 <= row < nrows - 1 and 1 <= col < ncols - 1
    
    def is_done(row, col, direction):
        """
        Checks to see whether or not any more tiles can be laid in the
        current direction given the buffer and the bounds.
        """
        for i in range(buffer):
            if direction == 0: # right
                if i != buffer - 1 and not in_bounds(row, col + i):
                    return True
                if (row, col + i) in route:
                    return True
            elif direction == 1: # down
                if i != buffer - 1 and not in_bounds(row + i, col):
                    return True
                if (row + i, col) in route:
                    return True
            elif direction == 2: # left
                if i != buffer - 1 and not in_bounds(row, col - i):
                    return True
                if (row, col - i) in route:
                    return True
            elif direction == 3: # up
                if i != buffer - 1 and not in_bounds(row - i, col):
                    return True
                if (row - i, col) in route:
                    return True
            else:
                raise ValueError("Invalid direction")
        return False
    
    direction = 0 # 0 = right, 1 = down, 2 = left, 3 = up
    black, path, start = [], [], (1, 1)
    row, col = start
    route = []
        
    while True:
        # generate a line of black squares until we run out of space
        if is_done(row, col, direction):
            break

        while in_bounds(row, col):
            # add current coordinates
            route.append((row, col))
            
            # proceed in the same direction if possible
            if direction == 0: # right
                col += 1
            elif direction == 1: # down
                row += 1
            elif direction == 2: # left
                col -= 1
            elif direction == 3: # up
                row -= 1
            else:
                raise Exception("Invalid direction")
            
            if is_done(row, col, direction):
                break

        direction = (direction + 1) % 4
        
    path = [coord for coord in route if coord[0] == 1]
    black = [coord for coord in route if coord[0] != 1]
    
    # uncomment to visualize
    # visualize_maze(map_builder(nrows, ncols, black, path, start))
    # plt.show()

    return (black, path, start), is_done
    

def generate_spiral_map_with_offshoots(nrows, ncols, base_path, is_done, interval_range=(5, 10), offshoot_length_range=(4, 8)):
    """
    Generates a map with the same path as the base path but with
    a few offshoot paths that don't lead to the goal.

    Args:
        nrows (int): number of rows in the map
        ncols (int): number of columns in the map
        base_path (tuple): contains the base path and start position
            in the form (black, path, start)
    """
    
    def get_offshoot_directions(row, col):
        """
        Given the proposed start point of an offshoot, return the possible
        directions for the offshoot to begin. 
        """
        candidates = []
        if col + 1 < ncols - 1 and (row, col + 1) not in route:
            candidates.append(0)
        if row + 1 < nrows - 1 and (row + 1, col) not in route:
            candidates.append(1)
        if col - 1 > 0 and (row, col - 1) not in route:
            candidates.append(2)
        if row - 1 >= 1 and (row - 1, col) not in route:
            candidates.append(3)
        return candidates

    def move_in_direction(row, col, direction):
        """
        Increments or decrements the row or column based on the direction.
        """
        if direction == 0: # right
            col += 1
        elif direction == 1: # down
            row += 1
        elif direction == 2: # left
            col -= 1
        elif direction == 3: # up
            row -= 1
        else:
            raise Exception("Invalid direction")
        return row, col
            
    
    black, path, start = base_path
    route = path + black
    steps_since_last_offshoot = 0

    for i in range(len(route)):
        if steps_since_last_offshoot >= random.randint(*interval_range):
            # choose length and direction of offshoot
            offshoot_length = random.randint(*offshoot_length_range)
            offshoot_directions = get_offshoot_directions(*route[i])
            
            # no possible offshoot here, so continue
            if len(offshoot_directions) == 0:
                continue
            
            # try all the offshoot candidates
            offshoot_candidates = []
            for direction in offshoot_directions:
                offshoot = []
                
                # initialize the offshoot in the chosen direction
                row, col = route[i]
                row, col = move_in_direction(row, col, direction)
                offshoot.append((row, col))
                row, col = move_in_direction(row, col, direction)
                offshoot.append((row, col))
            
                # extend the offshoot in a perpendicular direction
                direction = (direction + 1) % 4
                for j in range(offshoot_length):
                    row, col = move_in_direction(row, col, direction)
                    if (is_done(row, col, direction)) or j == offshoot_length - 1:
                        if len(offshoot) >= 3:
                            offshoot_candidates.append(offshoot)
                        break
                    offshoot.append((row, col))
            
            if len(offshoot_candidates) == 0:
                continue
            
            black.extend(random.choice(offshoot_candidates))
            steps_since_last_offshoot = 0
        
        steps_since_last_offshoot += 1
    
    # uncomment to visualize
    # visualize_maze(map_builder(nrows, ncols, black, path, start))
    # plt.show()
    
    return map_builder(nrows, ncols, black, path, start)

if __name__ == "__main__":
    # testing area
    rows, cols, buffer = 15, 15, 5
    interval_range = (5, 10)
    offshoot_length_range = (4, 8)
    # base_path, is_done = generate_spiral_base_path(rows, cols, buffer)
    # generate_spiral_map_with_offshoots(rows, cols, base_path, is_done, interval_range, offshoot_length_range)


    # map_1 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
    #       (3, 3, 3, 3, 0, 3, 0, 3, 3),
    #       (3, 3, 3, 3, 0, 3, 0, 3, 3),
    #       (3, 5, 6, 6, 6, 6, 6, 6, 3),
    #       (3, 6, 3, 3, 3, 3, 3, 6, 3),
    #       (3, 6, 6, 6, 6, 6, 6, 6, 3),
    #       (3, 3, 0, 0, 3, 3, 3, 3, 3),
    #       (3, 3, 3, 3, 3, 3, 3, 3, 3),)

    # new_size_maps = generate_different_size_maps(map_1)

    # map_visualizer(new_size_maps[2])