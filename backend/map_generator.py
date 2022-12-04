from mst_prototype import map_builder
import numpy as np
import copy

def generate_trick_maps(nmaps, nrows, ncols):
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
        trick_maps.append(map_builder(generate_trick_map(nrows, ncols, base_path)))
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

def generate_two_bigger_maps(map):
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
    """

    # Make a copy of the entire map and insert every other row or every other column

    twice_as_big_map = double_map_size(map)
    four_times_as_big_map = double_map_size(copy.deepcopy(twice_as_big_map))

    return [twice_as_big_map, four_times_as_big_map]


def generate_spiral_base_path(nrows, ncols):
    """
    Generates a spiral path that all trick maps will follow. 
    
    Args:
        nrows (int): number of rows in the map
        ncols (int): number of columns in the map
        
    Returns:
        A tuple containing
        black: list of tuples (int, int)
            a list of black squares
        path: list of tuples (int, int)
            a list of squares that form the path
        start: tuple
            the start position
    """
    def in_bounds(row, col):
        # buffer of one tile around the edge
        return 1 <= row < nrows - 1 and 1 <= col < ncols - 1
        
    
    direction = 0 # 0 = right, 1 = down, 2 = left, 3 = up
    done = False
    black, path, start = [], [], (1, 1)
    row, col = start
    buffer = 1
        
    while not done:
        # generate a line of black squares until we run out of space
        num_time_did_nothing_in_a_row = 0
        while in_bounds(row, col):
            # add current coordinates
            black.append((row, col))
            
            # proceed in the same direction if possible
            if direction == 0: # right
                col += 1
                for i in range(buffer):
                    if not in_bounds(row, col + i) or (row, col + i) in black:
                        break
            elif direction == 1: # down
                row += 1
                for i in range(buffer):
                    if not in_bounds(row + i, col) or (row + i, col) in black:
                        break
            elif direction == 2: # left
                col -= 1
                for i in range(buffer):
                    if not in_bounds(row, col - i) or (row, col - i) in black:
                        break
            elif direction == 3: # up
                row -= 1
                for i in range(buffer):
                    if not in_bounds(row - i, col) or (row - i, col) in black:
                        break
            else:
                raise Exception("Invalid direction")
            
        direction = (direction + 1) % 4
        
    print(black, path, start)
    return black, path, start
    
    

def generate_trick_map(nrows, ncols, base_path):
    pass

if __name__ == "__main__":
    # testing area
    generate_spiral_base_path(10, 10)