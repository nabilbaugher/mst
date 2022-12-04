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
    base_path = generate_base_path(nrows, ncols)
    trick_maps = []
    for _ in range(nmaps):
        trick_maps.append(generate_trick_map(nrows, ncols, base_path))
    return trick_maps

def generate_base_path(nrows, ncols):
    """
    Generates a path that all trick maps will follow.
    
    Args:
        nrows (int): number of rows in the map
        ncols (int): number of columns in the map
    """
    pass

def generate_trick_map(nrows, ncols, base_path):
    pass