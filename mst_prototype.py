
# !! NOTE !!
# This tree builder only works for paths of width 1

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import pprint

pp = pprint.PrettyPrinter(compact=False, width=90)

def memoize(function):
    """ ram cacher """
    memo = {}
    def wrapper(*args):
        if args not in memo:
            memo[args] = function(*args)
        return memo[args]
    return wrapper


# ----------------------
# Map Builder
# ----------------------

"""
A short summary of the various functions in this file:
    
    map_builder(nrows, ncols, black, path, start): return map_
        Uses a description of a map, and turns it into a "grid".
    
    raycaster(map_, pos): return observations
        Determines which tiles our player can see from their current position.
    
    new_observations(map_, pos): return new_observations
        Finds which tiles can the player see, which haven't been seen before
        
        *Uses raycaster
        
    update_map(map_, old_pos, new_pos): return map_updated
        Updates our grid map to reflect the player moving: updates black (unseen) tiles to path (seen)
        
        *Uses new_observations
        
    possible_paths(map_, pos): returns paths
        Gives us every possible path from the player's position to revealing a black tile.
        
    map2tree(map_): returns tree
        Converts our map into a tree of every possible path our player can take, assuming
        they always move to reveal black tiles.
        
        *Uses possible_paths, update_map
        
    map_visualizer(map_, node=None): returns None
        Takes in our map, and prints a version that is more human-readable
        
        *Uses map2tree, update_map
        
    softmax(values, tau): returns softvals
        Applies softmax to our values, with the parameter tau to make terms more/less similar.
    
    weight(p, beta): returns p_weighted
        Re-weights probabilities, using parameter beta so that 
        low probabilities are overestimated, and high probabilities are underestimated.
        

Overall representation:
    Our map is a tuple of tuples, where each tile is represented by a number:
        5 is the start tile
            Where player starts
            
        6 is the path tile
            Where player can walk, and is known by player
            
        0 is the black tile
            Unseen by player, converts into 6 when seen
            
        3 is the wall tile
            Cannot be walked on by player; blocks their vision from seeing other tiles
            
        There is no "exit" tile. In the real game, one black square hides an exit tile, but 
        in this game, our goal is to just find the value of our paths.
        
        We do not need to assign an exit tile to compute this value. 
        Instead, we assume an even probability of the exit being in any black square. 
    
"""

def map_builder(nrows, ncols, black, path, start):
    """
    This function turns a description of a map into its representation: a tuple of tuples, representing a grid.
    Each position on this grid is a "tile".
    
    Parameters
    ----------
    nrows : int. Number of rows
    ncols : int. Number of columns
    
    black : list of tuples (int, int)
        The tiles our player has not viewed yet.
    path :  list of tuples (int, int)
        The tiles which are part of our path.
    start : tuple (int, int). 
        Our starting position on the map.

    Returns
    -------
    tuple of tuples
        A representation of our map, where different values represent different tile types.

    """

    return tuple(
                tuple(     5 if (i ,j) == start  #Start tiles
                      else 6 if (i ,j) in path   #Path tiles
                      else 0 if (i ,j) in black  #Unseen tiles
                      else 3                     #Wall tiles (?)
                      for j in range(ncols)
                     )
                for i in range(nrows)
                )


# ----------------------
# Tree Builder
# ----------------------

def raycaster(map_, pos):
    """
    Based on our map and our current position, this function tells us which map tiles our player can see.
    
    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
           -tuple of tuples    has length = nrows, 
           -each tuple of ints has length = ncols.
           
    pos : tuple (int,int)
        The current position of our player on the map.

    Returns
    -------
    observed : set of tuples (int, int)
        Set of all tiles currently visible to the player.

    """

    (r, c)  = pos #Coords of player
    nrows, ncols = len(map_), len(map_[0]) #Shape of grid
    observed = set() #Tiles our player can see
    
    ###Our approach: start from our position (r,c). 
    ###Gradually move further away, check what we can see: walls block our vision.
    
    ##Break problem down by quadrant
    
    # 1st quadrant: +x, +y
    
    nearest_right_wall = ncols #Can't see past map
    
    #Limit to top half: [r, r-1, r-2, ..., 0]
    for r_ in range(r, -1, -1): 

        row_ = map_[r_] #Current row
        
        #Wall stops our player from seeing past it
        wall = min(nearest_right_wall, #Previous wall
                   row_.index(3, c)) #This row's closest wall: 3 is a wall
        
        #Both walls block: whichever wall is closer (min), will block more.
        
        #Limit to right half
        right = [c_ for c_ in range(c, wall )] #All of the points we can see: between our position and the wall
        observed.update([(r_, c_) for c_ in right])

        if not right: #If nothing new is seen, then walls are completely blocking our view.
            break

        nearest_right_wall = right[-1 ] +1 #Closest wall carries over as we move further away: still blocks

    # 2nd quadrant: -x, +y
    
    ##Getting left half by flipping map, and getting next "right" half
    nearest_left_wall = ncols
    
    for r_ in range(r, -1, -1): #Top half

        row_ = map_[r_][::-1] #Flip around our map: equivalent to x --> -x
        
        flipped_c = ncols - c #Player column in flipped map
        
        #Find closest wall to block
        wall = min(nearest_left_wall, 
                   row_.index(3, flipped_c-1)) 
        
        #Why flipped_c-1? Because flipped_c column is already handled by quadrant 1
        
        left = [c_ for c_ in range(flipped_c, wall)] #Visible tiles
        
        #n_col-c_ un-flips our coords
        observed.update([(r_, ncols-c_-1) for c_ in left]) 

        if not left: #Remaining squares blocked
            break

        nearest_left_wall = left[-1] + 1 #Remember wall

    # 3rd quadrant: -x, -y
    nearest_left_wall = ncols
    
    for r_ in range(r, nrows): #Bottom half

        row_ = map_[r_][::-1] #Flip map: x--> -x
        
        flipped_c = ncols - c
        
        wall = min(nearest_left_wall, 
                   row_.index(3, flipped_c -1))

        left = [c_ for c_ in range(flipped_c, wall)]
        
        #Unflip coords
        observed.update([(r_, ncols -c_ -1) for c_ in left])

        if not left:
            break

        nearest_left_wall = left[-1] + 1 

    # 4th quadrant: +x, -y
    nearest_right_wall = ncols
    
    for r_ in range(r, nrows): #Bottom half

        row_ = map_[r_] 
        
        wall = min(nearest_right_wall, 
                   row_.index(3, c))
        
        right = [c_ for c_ in range(c, wall)]
        observed.update([(r_, c_) for c_ in right])

        if not right:
            break

        nearest_right_wall = right[-1 ] +1

    #Result of all four quadrants
    return observed


def new_observations(map_, pos):
    """
    After taking a step, find which tiles are newly revealed to the player,
    as opposed to those they could already see.

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
           -tuple of tuples    has length = nrows, 
           -each tuple of ints has length = ncols.
           
    pos : tuple (int,int)
        The current position of our player on the map.

    Returns
    -------
    new_observations : set of tuples (int, int)
        All tiles the player can see from their current position.

    """

    observed = raycaster(map_, pos) #What our player can see now
    new_observations = set()

    for r ,c in observed:

        #0 represents "already seen by player"
        if map_[r][c] != 0: #If already seen, don't add
            continue

        new_observations.add((r ,c)) #If not seen, add to new 

    return new_observations


def update_map(map_, old_pos, new_pos):
    """
    

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
           -tuple of tuples    has length = nrows, 
           -each tuple of ints has length = ncols.
           
    old_pos : tuple (int,int) - previous position on map.
    new_pos : tuple (int,int) - current position on map.


    Returns
    -------
    map_updated: tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
           -tuple of tuples    has length = nrows, 
           -each tuple of ints has length = ncols.
           
        Map has been modified to account for the player having moved, 
        and the observations that come as a result.

    """

    observations = new_observations(map_, new_pos) #Get observations

    #Update map to reflect observations
    map_updated = [ 
                    [6 if (r ,c) in observations #A visible tile is now a "path" tile!
                     else map_[r][c] #Else, no change
                    for c in range(len(map_[0]))]
                    for r in range(len(map_))]

    map_updated[old_pos[0]][old_pos[1]] = 6 #Our old position is now a path tile: changes start position

    map_updated = tuple(tuple(row) for row in map_updated) #Convert to tuple of tuples

    return map_updated


def possible_paths(map_, pos):
    """
    Uses breadth-first search to find every possible movement to a black tile from our current position.
    
    Assumes we move towards some black tiles, and no backtracking.
    Thus, we only show paths that lead to us revealing black tiles. 
    
    Once any black tiles are revealed, the path terminates: we create a "partial path", that could be continued.
    
    We can think of this as every possible "next move" for our player, 
    assuming they're moving to a cluster of black tiles.

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
           -tuple of tuples    has length = nrows, 
           -each tuple of ints has length = ncols.
           
    pos : tuple (int,int)
        The current position of our player on the map.

    Returns
    -------
    paths : list of lists of tuples (int, int)
            -"List of lists" contains different possible paths
            -"List of tuples" is an array of positions along our path.
            
            For each path: the nth element of the list is the nth position on our path.
            
        Returns a list of all possible paths we could take, that reveal black tiles at the end. 
        These paths terminate upon reaching a black tile.
        

    """
    nrows, ncols = len(map_), len(map_[0])
    
    #BFS is implemented by the agenda. 
    agenda = [ [pos] ]
    paths = []

    while agenda: #Still agendas left - incomplete paths.

        path = agenda.pop(0) #Get top of agenda
        r_, c_ = path[-1] #Current position

        for rr, cc in ((0 ,1), (0 ,-1), (1 ,0), (-1 ,0)): #Possible movement directions
            
            within_bounds = lambda x, lower, upper: max(min(x,upper), lower)
        
            #Make sure r and c are between bounds
            r = within_bounds(r_ +rr,    0, nrows-1)
            c = within_bounds(c_ +cc,    0, ncols-1)

            # If this is a wall(3): can't move through
            # If in path: don't want to backtrack
            if map_[r][c] == 3 or (r, c) in path: #Don't include this direction!
                continue
            
            #Do we reveal a black square?
            observations = new_observations(map_, (r ,c))

            if observations: #If we reveal a black square, then we're finished.
                #Show the whole path, paired with the squares which are revealed.
                paths.append( ( path + [(r ,c)], observations ) )
                
            else: #No observation: path not ended, keep going.
                agenda.append(path + [(r ,c)])
                
    return paths



#Use memoize to cache results of map2tree, save for re-use
@memoize
def map2tree(map_):
    """
    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
           -tuple of tuples    has length = nrows, 
           -each tuple of ints has length = ncols.
           
    Returns
    -------
    tree: Dictionary of dictionaries - 
          -Outer dictionary - keys are int, representing the index of that node (a specific partial path)
                              vals are the node data as a dictionary
                              
          -Inner dictionary - keys are str representing node, vals give information about that particular node
          
        Stores all of the possible ways to move through the entire maze. Each node in our tree represents 
        a partial path.
    """
    
    pp.pprint(map_) #Print map
        
    # determine start position
    remains = 0
    for r, row in enumerate(map_):
        for c, val in enumerate(row):
            if val == 5: #Found start position!
                pos = (r ,c)
            elif val == 0: #Count the number of black tiles
                remains += 1
                
    #Create tree to edit
    
    #0 is the root node

    tree = {0: {'pos': pos,             #Player position: starting position
                'remains': remains,     #Count of black tiles not seen.
                'revealed': set(),      #Set of black tiles that have been revealed just now
                
                'path_from_par': [],    #Path from parent node
                'path_from_root': [],   #Path from root node
                'steps_from_par': 0,    #Steps from parent node
                'steps_from_root': 0,   #Steps from root node
                
                'children': set(),      #Child nodes: not set yet
                'pid': None}}           #Parent ID: the root node has no parent

    #BFS in progress
    agenda = [(0, map_)] #(node, current map)
    print('1', tree)
    
    while agenda: # in each loop, find and append children

        node, updated_map = agenda.pop(0) #Grab first element
        pos = tree[node]['pos'] #Current position
        
        print('PPPPP', possible_paths(updated_map, pos)) #What next moves are available?
        
        #Create a node for each next move
        #nobservation=new_observation output
        for path, nobservation in possible_paths(updated_map, pos): 
            branch = {'pos': path[-1],
                      'remains': tree[node]['remains' ] -len(nobservation),
                      'revealed': nobservation, #Which squares are revealed?
                      
                      'path_from_par': path, #Partial path
                      'path_from_root': tree[node]['path_from_root'] + path, #Full path
                      'steps_from_par': len(path) - 1, 
                      'steps_from_root': tree[node]['steps_from_root'] + len(path) - 1,
                      
                      'children': set(),
                      'pid': node, #Parent node id
                      'map': updated_map} #New modified map

            new_node = max(tree ) +1 #Enumerate: max gives us most recent (highest index) node
            
            #Add our new node to the agenda: expand this path further
            agenda.append((new_node, update_map(updated_map, path[0], path[-1])))
            
            #Make the parent-child relation
            tree[node]['children'].add(new_node)
            #Add child to tree
            tree[new_node] = branch
    print('t', tree)
    return tree


# ----------------------
# Map & Path Visualizer
# ----------------------
"""
0: hidden, 2: exit, 3: wall, 5: start, 6: open
"""

def map_visualizer(map_, node=None):
    """
    Turns a map representation into a human-interpretable image.

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
           -tuple of tuples    has length = nrows, 
           -each tuple of ints has length = ncols.
           
    node : int, optional
        Visualize a specific node for this map: in other words, show a partially explored map. 
        node is simply the number id for one of these partial paths.
            -Note: If the id is too high, there may be no corresponding node.

    Returns
    -------
    None
    
    
    Uses matplotlib.pyplot to make our map human-viewable.
    
    If node is given, the corresponding partially explored map will be displayed. Meaning, in this map,
    our player will have moved around to explore some of the black tiles.
    
    The map structure will still match map_, but it will display the path taken with a dotted line,
    And any squares that have been viewed by this player is 


    """

    nrows, ncols = len(map_), len(map_[0])
    
    #Using matplotlib.pyplot printing
    fig = plt.figure()
    ax = fig.add_subplot(111 ,aspect='equal')
    
    curr_map = map_
    
    if node: #If given a node, we've chosen a partial path.
             #Draw that path!
        
        tree = map2tree(map_) #Get tree to get nodes
        
        #Make sure that this node is valid!
        try:
            path = tree[node]['path_from_root']
        except:
            raise ValueError(f"The node value {node} is not in range: this node does not exist!")

        #Get the map matching this node
        curr_map = tree[node]['map']
        curr_pos = tree[node]['pos']
        prev_pos = path[-1]
        
        #Update map based on our last step
        curr_map = update_map(curr_map, curr_pos, prev_pos)
        
        #row --> y axis, col --> x axis
        #Thus, to do (x,y), we need tuples of the form (c,r)
        path = [(c ,r) for r ,c in path][::-1]
        #Also reversing order of path
        
        #Convert pairs of elements (x,y) into two lists: X and Y
        X, Y = zip(*[ (x + 0.5, nrows - y - 0.5)   for x ,y in path])
        #Offset (+0.5, -0.5) is so that our dots are centered on each tile
        
        
        ###Plotting our path
        
        #Draw dotted line between each tile on path
        ax.plot(X, Y, 'o--',  markersize=4, label=node)
        
        #Color our starting point (X[-1],Y[-1]) as purple
        ax.plot(X[-1], Y[-1], 's', markersize=8, color='purple')

    
    #Convert string numbers into int numbers
    curr_map = [[int(cell) for cell in row] for row in curr_map][::-1]
    #Convert tuples into lists so we can index
    
    #Old version: not sure why they made it more complicated??
    #curr_map = [[int(cell) for cell in list(row)[:ncols]] for row in curr_map][::-1]

    #Gather color map
    cmap = colors.ListedColormap \
        (['#9c9c9c', 'white', '#d074a4', '#b0943d', 'white', '#a1c38c', 'white', '#f5f5dc', 'moccasin'])
    boundaries = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=False)

    #Draw maze using a mesh
    ax.pcolormesh(curr_map, edgecolors='lightgrey', linewidth=1, cmap=cmap, norm=norm)
    ax.set_aspect('equal')

    # Major ticks positions
    #+.5 so they're centered on each square
    ax.set_xticks([ i +0.5 for i in range(ncols)])
    ax.set_yticks([ i +0.5 for i in range(nrows)[::-1]]) #Flip so it matches row/column counting

    # Major ticks label (for readability of plot, (0,0) at top left)
    ax.set_xticklabels([str(i) for i in range(ncols)])
    ax.set_yticklabels([str(i) for i in range(nrows)])

    plt.show()


# ----------------------
# Node Value Visualizer
# ----------------------

# parameters
TAUS    = np.linspace(0.1, 100, 20)
GAMMAS  = np.linspace(0 ,1 ,10)
BETAS   = np.linspace(0 ,2 ,10)
KAPPAS  = np.linspace(0 ,5 ,20)




def softmax(values, tau):
    """
    Applies the softmax function to our inputs. Tau allows us to emphasize or de-emphasize the
    difference between our values.

    Parameters
    ----------
    values : List of ints
        Set of values we want to take the softmax over.
        
    tau :    float
        Controls how "noisy" our softmax operation is: larger tau increases "noise",
        making our output terms more similar to each other.
        
        tau = 1: No effect. Softmax operates normally.
        tau > 1: Output terms more similar - closer to a uniform distribution.
        tau < 1: Output terms more different - emphasize larger values, downscale small values

    Returns
    -------
    list
        Returns the output of a softmax over our values.

    """

    numer = [np.exp(-v * ( 1 /tau)) for v in values]
    denom = sum(numer)
    return [ n /denom for n in numer]

""" probability weighting function: convert probability p to weight """

def weight(p, beta):
    """
    Re-weights probabilities, modelling on the human tendency 
    to overestimate small probabilities, and underestimate large probabilities.
    
    Parameters
    ----------
    p : float in real number range [0,1]
        Probability of an event.
        
    beta : float in real number range [0, inf]
        Represents the amount we follow the human pattern described above -
        bringing extreme values near 0 or 1 closer to the middle 0.5.
        
        beta = 1: No effect.
        beta > 1: Shows the effect desired: extremes move closer to 0.5.
        beta < 1: Opposite effect: extremes move further away from  0.5.

    Returns
    -------
    float
        Result of weighting: a probability in range [0,1]

    """
    
    return np.exp( -1 * (-np.log(p) )**beta )



""" return raw node value BEFORE softmax being applied """

###Rather than creating multiple different models, 
###the combined value model is used to generalize all three: EU, DU, PWU

def raw_nodevalue_comb(map_, node, gamma=1, beta=1):
    """
    Get value of this node (this path through our maze), 
    according to our parameters, before applying softmax and thus tau.

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
           -tuple of tuples    has length = nrows, 
           -each tuple of ints has length = ncols.
           
    node : int
        Node id - identifies which node you're identifying the value of.
        
        A node represents a partially explored map. 
        node is simply the number id for one of these partial paths.
            -Note: If the id is too high, there may be no corresponding node.
        
    gamma : float in real number range [0,1],  optional
        The "discount factor". Reflects the idea that we care less about rewards in the future.
        For every timestep in the future, we scale down the reward by a factor of gamma.
        
        Thus, larger gamma means we care more about future rewards. 
        Smaller gamma means we care less about future rewards.
        
    beta : float in real number range [0, inf], optional
        Represents the amount we follow the human pattern described above -
        bringing extreme values near 0 or 1 closer to the middle 0.5.
        
        beta = 1: No effect.
        beta > 1: Shows the effect desired: extremes move closer to 0.5.
        beta < 1: Opposite effect: extremes move further away from  0.5.

    Returns
    -------
    value : float
        Represents how "valuable" we think this particular path is, based on 
        what we know about possible futures, and our parameters.
        
        This does not take softmax into account.

    """

    tree = map2tree(map_)
    revealed = tree[node]["revealed"] #Get all of the black squares

    value, p_exit = 0, 0 #Initialize value as 0
    
    if tree[node]["pid"] != "NA": #NOT the root node: root node has no parent, no pid
        
        revealed_black = len(revealed) #Newly revealed tiles
        
        parent = tree[node]["pid"] #Get parent
        total_black = tree[parent]["remains"] #Get total number of black tiles
        
        #What's the chance of just having found the exit this turn?
        #Number of revealed tiles, divided by the tiles that remain.
        p_exit = revealed_black/total_black
        #Note that this is parent because we're ignoring any tiles from before this last turn
        #We already know those tiles weren't the exit, or the game would be over

        weighted_prob = weight(p_exit, beta) #Apply PWU: human bias in probabilities
        
        #Get distance to each possible exit: we assume one of them was correct!
        player_pos     = np.array(tree[node]["pos"])
        possible_exits = np.array(list(revealed))
        
        diff_to_exit   = np.abs( possible_exits - player_pos) #Difference in position, take abs
        dists_to_exit  = np.sum(diff_to_exit, axis=1) #Add x and y coords for manhattan dist
        
        #How many steps have we walked? How many steps will we walk to the exit?
        start_to_node = tree[node]["steps_from_root"] 
        node_to_goal  = np.mean( dists_to_exit ) #Average the distance to each exit 

        ###We originally had essentially
        ###node_to_goal = np.mean(list(revealed))
        ###Which doesn't seem to make any sense. Fixed?
        
        #Loss is the distance to the exit: add up two distance components.
        loss = start_to_node + node_to_goal
        
        #Current step value applied.
        value += weighted_prob * loss #Scale loss by probability
    
    #If we're at the root node, we haven't moved: there's no way that we already won.
    #Value for current step is 0.
        
    if tree[node].get("children", []): #Does this node have children?
        min_child_value = float("inf") #We want to pick min value: any will beat float("inf")

        for cid in tree[node]["children"]: #Iter over kids
            child_value = raw_nodevalue_comb(map_, cid, gamma, beta) #Do recursion
            
            if child_value < min_child_value: #Update val to find optimal child
                min_child_value = child_value
        
        weighted_comp = weight(1-p_exit, beta) #Re-weight the complement, (1-p)
        
        #Gamma is our discount factor: applied for future steps.
        value += gamma * weighted_comp * min_child_value
        #Future step value applied

    return value

########This old code is super confusing
########The function name ra_nodevalue is used literally nowhere else in the code
########The raw_nodevalue function makes no effect to include gamma or beta??
########Ngl I don't trust this

# def ra_nodevalue(maze, nid, gamma=1, beta=1):
    

#     tree = map2tree(maze)
#     cell_distances = tree[nid]["celldistances"]

#     value, p_exit = 0, 0

#     if tree[nid]["pid"] != "NA":

#         p_exit = len(cell_distances)/tree[tree[nid]["pid"]]["remains"]

#         value += weight(p_exit, beta) * (tree[nid]["steps_from_root"] + np.mean(list(cell_distances)))

#     if tree[nid].get("children", []):
#         min_child_value = float("inf")

#         for cid in tree[nid]["children"]:
#             child_value = raw_nodevalue(maze, cid, gamma, beta)
#             if child_value < min_child_value:
#                 min_child_value = child_value

#         value += gamma * weight(1-p_exit, beta) * min_child_value

#     return value

# def raw_nodevalue(maze, nid, gamma=1, beta=1):
#     """ return raw node value BEFORE softmax being applied """
#     Tree= map2tree(maze)
#     pid= Tree[nid]['pid']
#     return 1/len(Tree[pid]['children'])

def node_values(map_, parameters, raw_nodevalue_func=raw_nodevalue_comb):
    """
    Returns the value of every possible path (node) for the entire map.
    

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
           -tuple of tuples    has length = nrows, 
           -each tuple of ints has length = ncols.
           
    parameters : tuple of three floats.
                 parameters to help calculate values. 
                 
                 If using raw_nodevalue_func=raw_nodevalue_comb, 
                 Contains our parameters (tau, gamma, beta).
                 
    raw_nodevalue_func : function, optional
        A function that computes the value of a single node.
        Configurable, so we can try out different functions/parameters for computing node values.

    Returns
    -------
    values_summary : 
        Dictionary of all values based on node. 

    """

    values_summary = {} # {nid: {cid: value, cid: value, ...}}
    tree = map2tree(map_)

    for nid in tree:

        if nid == 'root': #No path: no need to find value
            continue

        children = tree[nid]['children']

        # ignore nid if it's not a decision node
        if len(children) <= 1:
            continue

        values_summary[nid] = {}
        print('Params', parameters)
        raw_values = [ raw_nodevalue_func(map_, cid, 1,1) for cid in children ]
        values = softmax(raw_values, 1)
        values_summary[nid][1] = {cid: val for cid ,val in zip(children, values)}

    return values_summary


def visualize_decision(maze, pid, ax):
    """
    0: hidden, 2: exit, 3: wall, 5: start, 6: open
    """
    tree = map2tree(maze)

    # custom color map
    cmap = colors.ListedColormap(['#9c9c9c', 'white', '#d074a4', '#b0943d', 'white', '#a1c38c', 'white', '#f5f5dc'])
    boundaries = [0, 1, 2, 3, 4, 5, 6, 7]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=False)

    # XXX fix orientation
    maze = [[int(cell) for cell in list(row)[:ncols]] for row in maze][::-1]

    # draw maze
    ax.pcolormesh(maze, edgecolors='lightgrey', linewidth=1, cmap=cmap, norm=norm)
    ax.set_aspect('equal')

    ax.set_xticks([0.5 + j for j in range(ncols)], minor=False)
    ax.set_xticklabels(list(range(ncols)))
    ax.set_yticks([0.5 + i for i in range(nrows)], minor=False)
    ax.set_yticklabels(list(range(nrows))[::-1])

    # plot path
    for nid in tree[pid]['children']:
        path = [(c ,r) for r ,c in tree[nid]['path_from_root']]
        x, y = zip(*[(x + 0.5, nrows - y - 0.5) for x ,y in path])
        ax.plot(x, y, 'o--',  markersize=4, label=nid)
        ax.plot(x[0], y[0], 's', markersize=8, color='purple')

    ax.legend(loc='upper left', bbox_to_anchor=(1 ,1))


def visualize_nodevalues(maze, pid, parameters, param_indx, model_name, raw_nodevalue_func, ax):

    tree = map2tree(maze)

    values_summary = node_values(maze, parameters, raw_nodevalue_func)

    decision_summary = {nid: [] for nid in tree[pid]['children']}

    for param in parameters:
        for nid, val in values_summary[pid][param].items():

            decision_summary[nid].append(val)

    for nid, values in decision_summary.items():
        ax.plot([param[param_indx] for param in parameters], values, 'o--', markersize=3, label=nid)

    ax.set_title(model_name)
    ax.grid()
    ax.legend()


def visualize_decision_and_nodevalues(maze, pid):

    _, axs = plt.subplots(2 ,2)
    axs = axs.flat

    # draw maze and decision paths
    visualize_decision(maze, pid, axs[0])

    # node value plot for each model

    parameters = [(tau ,1 ,1) for tau in TAUS]
    raw_nodevalue_func = raw_nodevalue
    # raw_nodevalue_func = random_nodevalue
    visualize_nodevalues(maze, pid, parameters, 0, 'expected utility', raw_nodevalue_func, axs[1])

    parameters = [(1, gamma, 1) for gamma in GAMMAS]
    raw_nodevalue_func = raw_nodevalue
    #raw_nodevalue_func = random_nodevalue
    visualize_nodevalues(maze, pid, parameters, 1, 'discounted utility', raw_nodevalue_func, axs[2])

    parameters = [(1, 1, beta) for beta in BETAS]
    raw_nodevalue_func = raw_nodevalue
    #raw_nodevalue_func = random_nodevalue
    visualize_nodevalues(maze, pid, parameters, 2, 'probability weighted utility', raw_nodevalue_func, axs[3])

    # parameters = [(tau,) for tau in TAUS]
    # raw_nodevalue_func = raw_nodevalue_h_cells
    # visualize_nodevalues(maze, pid, parameters, 0, 'cells', raw_nodevalue_func, axs[4])

    # parameters = [(tau,) for tau in TAUS]
    # raw_nodevalue_func = raw_nodevalue_h_steps
    # visualize_nodevalues(maze, pid, parameters, 0, 'steps', raw_nodevalue_func, axs[5])

    plt.show()




# if __name__ == "__main__":

#     import pprint
#     pp = pprint.PrettyPrinter(compact=False)

    # map 1
    # map_1 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
    #           (3, 3, 3, 3, 0, 3, 0, 3, 3),
    #           (3, 3, 3, 3, 0, 3, 0, 3, 3),
    #           (3, 5, 6, 6, 6, 6, 6, 6, 3),
    #           (3, 6, 3, 3, 3, 3, 3, 6, 3),
    #           (3, 6, 6, 6, 6, 6, 6, 6, 3),
    #           (3, 3, 0, 0, 3, 3, 3, 3, 3),
    #           (3, 3, 3, 3, 3, 3, 3, 3, 3),)

#     ncols, nrows = 13, 9

#     # map 2
#     ncols, nrows = 13, 9
#     start = (5 ,4)

#     path = {(3 ,1), (3 ,2), (3 ,3), (3 ,4), (3 ,5), (3 ,6), (3 ,7), (3 ,8),
#             (5 ,1), (5 ,2), (5 ,3), (5 ,4), (5 ,5), (5 ,6), (5 ,7), (5 ,8),
#             (4 ,1), (4 ,8)}

#     black = {(6 ,2), (7 ,2),
#              (6 ,7), (7 ,7),
#              (2 ,4), (1 ,4), (1 ,5), (1 ,6),
#              (4 ,9), (4 ,10), (4 ,11)}

#     map_2 = map_builder(nrows, ncols, black, path, start)

#     # map 3
#     # ncols, nrows = 12, 8
#     # start = (1,3)

#     # path = {(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10),
#     #         (2,3), (3,3), (4,3), (5,3), (6,3)}

#     # black = {(2,1), (3,1),
#     #          (5,4), (5,5), (5,6), (6,4), (6,5), (6,6),
#     #          (2,9), (2,10), (3,9), (3,10), (4,9), (4,10), (5,9), (5,10), (6,9)}

#     # map_3 = map_builder(nrows, ncols, black, path, start)

#     # map 4
#     # ncols, nrows = 12, 14
#     # start = (7,3)

#     # path = {(6,3), (5,3), (4,3), (3,3), (3,4), (3,5), (3,6), (3,6), (3,7), (3,8), (3,9), (3,10),
#     #         (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7), (7,8), (7,9), (7,10),
#     #         (8,3), (9,3), (10,3), (11,3)}

#     # black = {(1,7), (1,8), (1,9), (1,10), (2,7), (2,8), (2,9), (2,10), (4,7), (4,8), (4,9), (4,10), (5,7), (5,8), (5,9), (5,10),
#     #          (8,1), (9,1),
#     #          (11,4), (11,5), (11,6), (11,7),
#     #          (8,9), (8,10), (9,9), (9,10), (10,9), (10,10), (11,9), (11,10)}

#     # map_4 = map_builder(nrows, ncols, black, path, start)

#     # map 5
#     # ncols, nrows = 13, 5
#     # start = (1,8)

#     # path = {(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), (1,11)}

#     # black = {(2,1), (2,2), (2,3), (3,1), (3,2), (3,3),
#     #          (2,11), (3,11)}

#     # map_5 = map_builder(nrows, ncols, black, path, start)

#     # map 6
#     # ncols, nrows = 14, 11
#     # start = (6,1)

#     # path = {(6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8), (6,9), (6,10), (6,11), (6,12),
#     #         (7,1), (8,1), (5,3), (4,3), (3,3), (6,5), (7,5), (8,5), (9,5),
#     #         (6,7), (5,7), (4,7), (3,7), (2,7), (1,7)}

#     # black = {(8,2), (8,3), (4,4), (4,5), (3,4), (3,5), (8,6), (8,7), (8,8), (8,9), (9,6), (9,7), (9,8), (9,9),
#     #          (1,8), (1,9), (1,10), (1,11), (2,8), (2,9), (2,10), (2,11),
#     #          (3,8), (3,9), (3,10), (3,11), (4,8), (4,9), (4,10), (4,11),}

#     # map_6 = map_builder(nrows, ncols, black, path, start)

#     # map 7
#     # ncols, nrows = 14, 11
#     # start = (6,1)

#     # path = {(6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8), (6,9), (6,10), (6,11), (6,12),
#     #         (7,1), (8,1), (5,3), (4,3), (3,3), (6,5), (7,5), (8,5), (9,5),
#     #         (6,7), (5,7), (4,7), (3,7), (2,7), (1,7)}

#     # black = {(8,2), (8,3), (4,4), (4,5), (3,4), (3,5), (8,6), (8,7), (8,8), (9,6), (9,7), (9,8),
#     #          (3,8), (3,9), (3,10), (3,11), (4,8), (4,9), (4,10), (4,11),}

#     # map_7 = map_builder(nrows, ncols, black, path, start)

#     # map 8
#     # ncols, nrows = 13, 10
#     # start = (6,1)

#     # path = {(8,1), (7,1), (6,1), (5,1), (4,1), (3,1), (2,1),
#     #         (5,2), (5,3), (5,4),
#     #         (2,4), (3,4), (4,4), (6,4), (7,4), (8,4),
#     #         (8,5), (8,6),
#     #         (7,6), (6,6), (5,6),
#     #         (5,7), (5,7), (5,8), (5,9), (5,10), (5,11),
#     #         (2,5), (2,6), (2,7), (2,8), (2,9), (2,10)}

#     # black = {(8,2), (2,2), (2,3), (3,3),
#     #          (1,6), (1,7), (1,8), (1,9), (1,10), (3,6), (3,7), (3,8), (3,9), (3,10),
#     #          (6,8), (6,9), (6,10), (6,11),
#     #          (7,8), (7,9), (7,10), (7,11),
#     #          (8,8), (8,9), (8,10), (8,11),}

#     # map_8 = map_builder(nrows, ncols, black, path, start)

#     # --------------------------------------------------------------------

#     visualize_decision_and_nodevalues(map_1, 0)

map_1 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
          (3, 3, 3, 3, 0, 3, 0, 3, 3),
          (3, 3, 3, 3, 0, 3, 0, 3, 3),
          (3, 5, 6, 6, 6, 6, 6, 6, 3),
          (3, 6, 3, 3, 3, 3, 3, 6, 3),
          (3, 6, 6, 6, 6, 6, 6, 6, 3),
          (3, 3, 0, 0, 3, 3, 3, 3, 3),
          (3, 3, 3, 3, 3, 3, 3, 3, 3),)
