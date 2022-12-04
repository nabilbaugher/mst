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
        A tuple of tuples (like the return of map_builder).
    """
    # just trying spiral for now
    base_path = generate_spiral_base_path(nrows, ncols)
    trick_maps = []
    for _ in range(nmaps):
        trick_maps.append(map_builder(generate_trick_map(nrows, ncols, base_path)))
    return trick_maps

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