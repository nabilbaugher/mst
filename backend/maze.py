import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import pprint
from collections import deque


class Maze:
    
    def __init__(self, nrows, ncols, black, path, start, exit_=None):
        
        self.nrows, self.ncols = nrows, ncols
        self.black, self.path, self.start, self.exit = black, path, start, exit_
        
        self.map_builder() #Build map representation
        
    def __str__(self):
        return str(self.map)
    
    def __repr__(self):
        return repr(self.map)
    
    def __getitem__(self, index):
        return self.map[index]
    
    def __len__(self):
        return len(self.map)


    def map_builder(self):
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
        exit_:  tuple (int, int). Optional
            The "win condition": if the player moves to this tile, the game ends. 
            If this parameter is set to None, there is no exit. If it is set to 'rand', it will randomly select from the blacks
    
        Any square not specified becomes a wall square, which the player cannot traverse.
        
        Returns
        -------
        tuple of tuples
            A representation of our map, where different values represent different tile types.
    
        """
        #if exit == None, then we have no exit
            
        if self.exit == 'rand': #Randomly generate our exit
            self.exit = random.choice(self.black)
            
        
    
        map_ = tuple(
                    tuple(     5 if (i ,j) == self.start  #Start tiles
                          else 6 if (i ,j) in self.path   #Path tiles
                          else 2 if (i ,j) == self.exit  #Exit tile
                          else 0 if (i ,j) in self.black  #Unseen tiles
                          else 3                     #Wall tiles (?)
                          for j in range(self.ncols)
                         )
                    for i in range(self.nrows)
                    )
        
        self.map = map_ #Save map representation
        

    def raycaster(self, pos):
        """
        Based on our map and our current position, this function tells us which map tiles our player can see.
        
        Parameters
        ----------
        self.map : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
               Our "maze" the player is moving through.
               
        pos : tuple (int,int)
            The current position of our player on the map.
    
        Returns
        -------
        observed : set of tuples (int, int)
            Set of all tiles currently visible to the player.
    
        """
    
        (r, c)  = pos #Coords of player
        observed = set() #Tiles our player can see
        
        ###Our approach: start from our position (r,c). 
        ###Gradually move further away, check what we can see: walls block our vision.
        
        ##Break problem down by quadrant
        
        # 1st quadrant: +x, +y
        
        nearest_right_wall = self.ncols #Can't see past map
        
        #Limit to top half: [r, r-1, r-2, ..., 0]
        for r_ in range(r, -1, -1): 
    
            row_ = self.map[r_] #Current row
            
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
        nearest_left_wall = self.ncols
        
        for r_ in range(r, -1, -1): #Top half
    
            row_ = self.map[r_][::-1] #Flip around our map: equivalent to x --> -x
            
            flipped_c = self.ncols - c #Player column in flipped map
            
            #Find closest wall to block
            wall = min(nearest_left_wall, 
                       row_.index(3, flipped_c-1)) 
            
            #Why flipped_c-1? Because flipped_c column is already handled by quadrant 1
            
            left = [c_ for c_ in range(flipped_c, wall)] #Visible tiles
            
            #n_col-c_ un-flips our coords
            observed.update([(r_, self.ncols-c_-1) for c_ in left]) 
    
            if not left: #Remaining squares blocked
                break
    
            nearest_left_wall = left[-1] + 1 #Remember wall
    
        # 3rd quadrant: -x, -y
        nearest_left_wall = self.ncols
        
        for r_ in range(r, self.nrows): #Bottom half
    
            row_ = self.map[r_][::-1] #Flip map: x--> -x
            
            flipped_c = self.ncols - c
            
            wall = min(nearest_left_wall, 
                       row_.index(3, flipped_c -1))
    
            left = [c_ for c_ in range(flipped_c, wall)]
            
            #Unflip coords
            observed.update([(r_, self.ncols -c_ -1) for c_ in left])
    
            if not left:
                break
    
            nearest_left_wall = left[-1] + 1 
    
        # 4th quadrant: +x, -y
        nearest_right_wall = self.ncols
        
        for r_ in range(r, self.nrows): #Bottom half
    
            row_ = self.map[r_] 
            
            wall = min(nearest_right_wall, 
                       row_.index(3, c))
            
            right = [c_ for c_ in range(c, wall)]
            observed.update([(r_, c_) for c_ in right])
    
            if not right:
                break
    
            nearest_right_wall = right[-1 ] +1
    
        #Result of all four quadrants
        return observed
    
        
    def new_observations(self, pos):
        """
        After taking a step, find which tiles are newly revealed to the player,
        as opposed to those they could already see.
    
        Parameters
        ----------
        self.map : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
               Our "maze" the player is moving through.
               
        pos : tuple (int,int)
            The current position of our player on the map.
    
        Returns
        -------
        new_observations : set of tuples (int, int)
            All tiles the player can see from their current position.
    
        """
    
        observed = self.raycaster(pos) #What our player can see now
        new_observations = set()
    
        for r ,c in observed:
    
            #0 represents "already seen by player"
            if self.map[r][c] not in (0,2): #If already seen, don't add
                continue
    
            new_observations.add((r ,c)) #If not seen, add to new 
    
        return new_observations
    
    def update_map(self, old_pos, new_pos):
        """
        

        Parameters
        ----------
        self.map : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
               Our "maze" the player is moving through.
               
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

        nobservations = self.new_observations(new_pos) #Get observations
        
        new_path = self.path.copy()
        
        new_path+= list(nobservations)
        new_path+= old_pos #Include old position

        new_maze = Maze(self.nrows, self.ncols, self.black, new_path, self.start, self.exit)
        

        return new_maze
    
    
    def possible_paths(self, pos):
        """
        Uses breadth-first search to find every possible movement to a black tile from our current position.
        
        Assumes we move towards some black tiles, and no backtracking.
        Thus, we only show paths that lead to us revealing black tiles. 
        
        Once any black tiles are revealed, the path terminates: we create a "partial path", that could be continued.
        
        We can think of this as every possible "next move" for our player, 
        assuming they're moving to a cluster of black tiles.
    
        Parameters
        ----------
        self.map : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
               Our "maze" the player is moving through.
               
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
        
        #BFS is implemented by the agenda. 
        # agenda = [ [pos] ]
        agenda = deque([[pos]])
        paths = []
    
        while agenda: #Still agendas left - incomplete paths.
            path = agenda.popleft() #Get top of agenda
            r_, c_ = path[-1] #Current position
    
            for rr, cc in ((0 ,1), (0 ,-1), (1 ,0), (-1 ,0)): #Possible movement directions
                
                within_bounds = lambda x, lower, upper: max(min(x,upper), lower)
            
                #Make sure r and c are between bounds
                r = within_bounds(r_ +rr,    0, self.nrows-1)
                c = within_bounds(c_ +cc,    0, self.ncols-1)
    
                # If this is a wall(3): can't move through
                # If in path: don't want to backtrack
                if self.map[r][c] == 3 or (r, c) in path: #Don't include this direction!
                    continue
                
                #Do we reveal a black square?
                observations = self.new_observations( (r ,c) )
    
                if observations: #If we reveal a black square, then we're finished.
                    #Show the whole path, paired with the squares which are revealed.
                    paths.append( ( path + [(r ,c)], observations ) )
                    
                else: #No observation: path not ended, keep going.
                    agenda.append(path + [(r ,c)])

                    
        return paths
    
    def visualize(self, node=None):
        """
        Turns a map representation into a human-interpretable image.

        Parameters
        ----------
        self.map  : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
                
                Our "maze" the player is moving through.
               
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
        
        #Using matplotlib.pyplot printing
        fig = plt.figure()
        ax = fig.add_subplot(111 ,aspect='equal')
        
        curr_map = self
        
        if node: #If given a node, we've chosen a partial path.
                  #Draw that path!
            
            tree = maze2tree(self) #Get tree to get nodes
            
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
            curr_map = curr_map.update_map(curr_pos, prev_pos)
            
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
        ax.set_xticks([ i +0.5 for i in range(self.ncols)])
        ax.set_yticks([ i +0.5 for i in range(self.nrows)[::-1]]) #Flip so it matches row/column counting

        # Major ticks label (for readability of plot, (0,0) at top left)
        ax.set_xticklabels([str(i) for i in range(self.ncols)])
        ax.set_yticklabels([str(i) for i in range(self.nrows)])

        plt.show()

###Used so we don't have to re-calculate the same tree multiple times
def memoize(function):
    """ ram cacher """
    memo = {}
    def wrapper(*args):
        if args not in memo:
            memo[args] = function(*args)
        return memo[args]
    return wrapper

#Use memoize to cache results of map2tree, save for re-use

@memoize
def maze2tree(maze):
    """
    Parameters
    ----------
    maze : Maze object, our maze to navigate. Stores a maze.map object:
        
        map: tuple of tuples of ints - represents a grid in the shape (maze.nrows, maze.ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
               Our "maze" the player is moving through.
           
    Returns
    -------
    tree: Dictionary of dictionaries - 
          -Outer dictionary - keys are int, representing the index of that node (a specific partial path)
                              vals are the node data as a dictionary
                              
          -Inner dictionary - keys are str representing node, vals give information about that particular node
          
        Stores all of the possible ways to move through the entire maze. Each node in our tree represents 
        a partial path.
    """
    if type(maze)==tuple: #If we have a grid
        maze = Maze(maze)
    
    #pp.pprint(map_) #Print map
        
    # determine start position
    remains = 0
    for r, row in enumerate(maze.map):
        for c, val in enumerate(row):
            if val == 5: #Found start position!
                pos = (r ,c)
            elif val in (0,2): #Count the number of black tiles
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
    # agenda = [(0, map_)] #(node, current map)
    agenda = deque([(0, maze)]) #(node, current map)
    #print('1', tree)
    
    while agenda: # in each loop, find and append children

        node, new_map = agenda.popleft() #Grab first element
        pos = tree[node]['pos'] #Current position
        
        #print('PPPPP', possible_paths(updated_map, pos)) #What next moves are available?
        
        #Create a node for each next move
        #nobservation=new_observation output
        for path, nobservation in new_map.possible_paths(pos): 
            branch = {'pos': path[-1],
                      'remains': tree[node]['remains' ] -len(nobservation),
                      'revealed': nobservation, #Which squares are revealed?
                      
                      'path_from_par': path, #Partial path
                      'path_from_root': tree[node]['path_from_root'] + path, #Full path
                      'steps_from_par': len(path) - 1, 
                      'steps_from_root': tree[node]['steps_from_root'] + len(path) - 1,
                      
                      'children': set(),
                      'pid': node, #Parent node id
                      'map': new_map.map} #New modified map

            new_node = max(tree ) +1 #Enumerate: max gives us most recent (highest index) node
            
            #Add our new node to the agenda: expand this path further
            
            agenda.append((new_node, new_map.update_map(path[0], path[-1])))
            
            #Make the parent-child relation
            tree[node]['children'].add(new_node)
            #Add child to tree
            tree[new_node] = branch
            
        if len(agenda)>100:
            print(new_map, path[-1])
            
    #print('t', tree)
    return tree

####################################
#Functions used when we don't want to deal with mazes, just grids

def map_builder(nrows, ncols, black, path, start, exit_=None):
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
    exit_:  tuple (int, int). Optional
        The "win condition": if the player moves to this tile, the game ends. 
        If this parameter is set to None, there is no exit. If it is set to 'rand', it will randomly select from the blacks

    Any square not specified becomes a wall square, which the player cannot traverse.
    
    Returns
    -------
    tuple of tuples
        A representation of our map, where different values represent different tile types.

    """
    maze = Maze(nrows, ncols, black, path, start, exit_ )
    return maze.map

def map_visualizer(map_, node=None):
    """
    Turns a map representation into a human-interpretable image.

    Parameters
    ----------
    map_ : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
           -tuple of tuples    has length = nrows, 
           -each tuple of ints has length = ncols.
            
            Our "maze" the player is moving through.
           
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
    maze = grid2maze(map_)
    maze.visualize(node)
    


def grid2maze(map_):
    nrows, ncols = len(map_), len(map_[0])
    
    black, path, start, exit_ = [],[], None, None
    
    for r in range(nrows):
        for c in range(ncols):
            if map_[r][c]==0:
                black.append( (r,c) )
            elif map_[r][c]==6:
                path.append( (r,c) )
            elif map_[r][c]==2:
                exit_=(r,c)
            elif map_[r][c]==5:
                start=(r,c)
                
    return Maze(nrows, ncols, black, path, start, exit_)
            