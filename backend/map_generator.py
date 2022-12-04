import matplotlib.pyplot as plt
from mst_prototype import map_builder
from simulation import visualize_maze

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
                if (row, col + i) in path_and_black:
                    return True
            elif direction == 1: # down
                if i != buffer - 1 and not in_bounds(row + i, col):
                    return True
                if (row + i, col) in path_and_black:
                    return True
            elif direction == 2: # left
                if i != buffer - 1 and not in_bounds(row, col - i):
                    return True
                if (row, col - i) in path_and_black:
                    return True
            elif direction == 3: # up
                if i != buffer - 1 and not in_bounds(row - i, col):
                    return True
                if (row - i, col) in path_and_black:
                    return True
            else:
                raise ValueError("Invalid direction")
        return False
    
    direction = 0 # 0 = right, 1 = down, 2 = left, 3 = up
    black, path, start = [], [], (1, 1)
    row, col = start
    path_and_black = []
        
    while True:
        # generate a line of black squares until we run out of space
        if is_done(row, col, direction):
            break

        while in_bounds(row, col):
            # add current coordinates
            path_and_black.append((row, col))
            
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
        
    path = [coord for coord in path_and_black if coord[0] == 1]
    black = [coord for coord in path_and_black if coord[0] != 1]
    
    # # uncomment to visualize
    # visualize_maze(map_builder(nrows, ncols, black, path, start))
    # plt.show()

    return black, path, start
    

def generate_spiral_map_with_offshoots(nrows, ncols, base_path, interval_range=(5, 10), offshoot_length_range=(4, 8)):
    """
    Generates a map with the same path as the base path but with
    a few offshoot paths that don't lead to the goal.

    Args:
        nrows (int): number of rows in the map
        ncols (int): number of columns in the map
        base_path (tuple): contains the base path and start position
            in the form (black, path, start)
    """

if __name__ == "__main__":
    # testing area
    generate_spiral_base_path(15, 15)