from mst_prototype import map_builder

def generate_trick_maps(nmaps, nrows, ncols):
    """
    Generates a set of trick maps that all follow the same path to the goal
    but have different extraneous paths.
    
    Args:
        nmaps (int): number of maps to generate
        nrows (int): number of rows in the map
        ncols (int): number of columns in the map
    
    Returns:
        A list of maps where each map is represented as the arguments to
        map_builder() which are nrows, ncols, black, path, start.
    """
    # just trying spiral for now
    base_path = generate_spiral_base_path(nrows, ncols)
    trick_maps = []
    for _ in range(nmaps):
        trick_maps.append(generate_trick_map(nrows, ncols, base_path))
    return trick_maps

def generate_two_bigger_maps(map):
    """
    Given a map with nrows rows and ncolumns columns, create a new map that is a different size

    To keep it simple, we are only generating 
        * 1 map that is double the size of the original map
        * 1 map that is 4 times the size of the original map

    We are making a bigger map by copying every column and row
        (Except if a row or column contains 5, the starting space)

    Args:
        map (tuple of tuples): A grid where the number in each space represents:
        * 5 is the start tile
        * 6 is the path tile
        * 0 is the black tile
        * 3 is the wall tile

    Returns:
        A list of maps where each map is a tuple of tuples
    """

    nrows = len(map)
    ncols = len(map[0])



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
        return 0 <= row < nrows and 0 <= col < ncols
    
    direction = 0 # 0 = right, 1 = down, 2 = left, 3 = up
    done = False
    
    # sus code do not read pls | haha bet
    while not done:
        black = []
        path = []
        start = (0, 0)
        row = 0
        col = 0
        while in_bounds(row, col):
            print(path)
            path.append((row, col))
            if direction == 0:
                col += 1
            elif direction == 1:
                row += 1
            elif direction == 2:
                col -= 1
            elif direction == 3:
                row -= 1
            else:
                raise Exception("Invalid direction")
            direction = (direction + 1) % 4
        if len(path) > 1:
            break
    print(black, path, start)
    return black, path, start
    
    

def generate_trick_map(nrows, ncols, base_path):
    pass

if __name__ == "__main__":
    # testing area
    generate_spiral_base_path(10, 10)