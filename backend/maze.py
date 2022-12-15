import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import pprint
from collections import deque





"""
The goal of this file is to build a Maze representation that can be used by other files. This creates an 
environment in which we can do the Maze Search Task.

A maze is a useful and simplified way to observe how humans make decisions and judgements. This file creates 
a system the subjects are able to interact with, and the experimenters can analyze.

========================================================

class Maze():
    The class representing our maze: takes our base grid, and adds some functionality to it.
    
    get_hidden(self): return hidden
        Get all the squares that our player hasn't seen on this map.
    
    map_builder(self): return None
        Uses a description of a map, and turns it into a "grid".
        
    raycaster(self, pos): return observations
        Determines which tiles our player can see from their current position.
        
    new_observations(self, pos): return new_observations
        Finds which tiles can the player see, which haven't been seen before
    
        *Uses self.raycaster
        
    update_map(self, old_pos, new_pos): return map_updated
        Updates our grid map to reflect the player moving: updates black (unseen) tiles to path (seen)
        
        *Uses self.new_observations
        
    possible_paths(self, pos): returns paths
        Gives us every possible path from the player's position to revealing a black tile.
        
        *Uses self.new_observations
        
    visualize(self, node=None): returns None
        Takes in our map, and prints a version that is more human-readable.
        
    

A short summary of the various functions in this file:
    
    
    DEPRECATED --- maze2tree(maze): returns tree
        Converts our map into a tree of every possible path our player can take, assuming
        they always move to reveal black tiles.
        
        *Uses maze.possible_paths, maze.update_map
        
        ##Used so we can properly take advantage of memoizing
    
    maze2graph(maze): return graph
        Converts our map into a graph, representing all of the choices we could make as we move through
        the maze.

        



Overall representation:
    Our map is a tuple of tuples, where each tile is represented by a number:
        5 is the start tile
            Where player starts
            
        6 is the path tile
            Where player can walk, and is known by player
            
        0 is the black tile
            Unseen by player, converts into 6 when seen.
            
        3 is the wall tile
            Cannot be walked on by player; blocks their vision from seeing other tiles
        
        2 is the exit tile
            Looks like a black tile until observed by the player: reaching it is the win condition
        
        When calculating, we assume that the exit tile is equally likely to be in any of the seemingly black squares,
        even if we actually know where the exit tile is.
    
    
    
    
Reminder of the params typical format
    parent_params_comb = ('tau',)
    node_params_comb = ('gamma','beta')
"""





class Maze:
    
    def __init__(self, nrows, ncols, black, path, start, 
                 exit_ = None, pos = None, name = None):
        
        self.nrows, self.ncols = nrows, ncols
        self.black, self.path, self.start, self.exit = black, path, start, exit_
        
        self.pos, self.name = pos, name
        
        if self.pos==None: #Position stored internal to the maze
            self.pos=start

            
        self.map_builder() #Build map representation
        
        
            
        
    def __str__(self):
        string='Maze '
        if self.name!=None:
            string+=self.name
        else:
            string+='Unnamed'
            
        string+="_position_"+str(self.pos)
        
        return string
    
    def __repr__(self):
        return str(self)
    
    def __getitem__(self, index):
        return self.map[index]
    
    def __iter__(self):
        for row in self.map:
            yield(row)
    
    def __len__(self):
        return len(self.map)
    
    def __eq__(self, other):
        """
        A maze is uniquely identified by its map, and your current position.
        """
        
        if str(type(self)) != str(type(other)):
            return False
        
        return self.map == other.map and self.pos == other.pos
    
    def __hash__(self):
        return hash( (self.map, self.pos) )

    def get_hidden(self):
        """Gets all hidden squares there are, as a list."""
        
        if 'hidden' not in self.__dict__: #Generate our hiddens if we haven't done so before
            
            hidden = self.black.copy() #Black squares
            
            if self.exit!=None: #Exit square exists
                hidden.append(self.exit)
            
            self.hidden = tuple(hidden) #Makes it hashable
        
        return self.hidden
            
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
        Based on our map and a new position, this function tells us which map tiles our player can see.
        
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
        
        def find_wall(row, lower):
            """ Find the nearest wall to the right. """
            if 3 not in row[lower:]:
                return float('inf') #No wall!
            
            return row.index(3,lower) #If there's a wall, find the closest one.
            
        
        # 1st quadrant: +x, +y
        
        nearest_right_wall = self.ncols #Can't see past map
        
        #Limit to top half: [r, r-1, r-2, ..., 0]
        for r_ in range(r, -1, -1): 
    
            row_ = self.map[r_] #Current row
            
            #Wall stops our player from seeing past it
            wall = min(nearest_right_wall, #Previous wall
                       find_wall(row_,c)) #This row's closest wall: 3 is a wall
            #c: Starting at the point (r,c), moving to the right
            
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
                       find_wall(row_,flipped_c-1)) 
            
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
                       find_wall(row_,flipped_c-1))
    
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
                       find_wall(row_,c))
            
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
    
    def valid_pos(self, pos):
        """
        Check whether a given position is valid on this map.
        """
        r,c = pos
        
        #Within bounds
        r_valid = (r>=0) and (r<=self.nrows-1)
        c_valid = (c>=0) and (c<=self.ncols-1)
        
        if not r_valid or not c_valid: return False #Both conditions must be met to be in bounds
        
        #If it's in bounds, we have to check if it's within a wall
        not_in_wall = (self.map[r][c]!=3)
        
        return not_in_wall #All three must be met
    
    def move(self, pos, shift):
        
        
        """
        Try to move on the map.
        
        Parameters
        ----------
        shift : tuple (int,int) 
            How we move from our old position.
            
        Returns
        -------
        pos: if we can move, it returns new_pos. else, we stay in the same place. 
        
        Raise error if old_pos is invalid.
        """
        r,c   =   pos
        rr,cc   = shift #Move positions
        
        new_pos = ( r+rr, c+cc )
        
        if not self.valid_pos(pos): #We need to at least start on a valid square!
            raise ValueError("Original position is not valid! We cannot be standing on this square.")
        
        if not self.valid_pos(new_pos): #If new square isn't valid, stay on old square.
            return pos
        
        return new_pos #Otherwise, move!
        
    def update_map(self, pos):
        """
        

        Parameters
        ----------
        self.map : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
               Our "maze" the player is moving through.
               
        pos : tuple (int,int) - new position on map.


        Returns
        -------
        map_updated: tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
            Map has been modified to account for the player having moved, 
            and the observations that come as a result.

        """

        nobservations = self.new_observations(pos) #Get observations
        
        new_path = self.path.copy()
        
        new_path+= list(nobservations)
        
        new_black = [black for black in self.black if black not in nobservations] #Remove old black
        
        if self.exit in nobservations:
            new_exit=None
        else:
            new_exit=self.exit

        new_maze = Maze(self.nrows, self.ncols, new_black, new_path, self.start, 
                        pos=pos, exit_=new_exit, name=self.name)
        

        return new_maze
       
    def possible_paths(self):
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
        agenda = deque([[self.pos]])
        paths = []
    
        while agenda: #Still agendas left - incomplete paths.
            path = agenda.popleft() #Get top of agenda
            curr_pos = path[-1] #Current position
    
            for shift in ((0 ,1), (0 ,-1), (1 ,0), (-1 ,0)): #Possible movement directions
                
                #Try to move in this direction
                new_pos = self.move(curr_pos,shift)
    
                # If in path: don't want to backtrack
                
                if new_pos in path: #Don't include this direction!
                    continue
                
                #Do we reveal a black square?
                observations = self.new_observations( new_pos )
    
                if observations: #If we reveal a black square, then we're finished.
                    #Show the whole path, paired with the squares which are revealed.
                    paths.append( path + [new_pos])
                    
                else: #No observation: path not ended, keep going.
                    agenda.append(path + [new_pos])

                    
        return paths
    
    def visualize(self, ax=None, pos=None, path=None):
        """
        Turns a map representation into a human-interpretable image.

        Parameters
        ----------
        self.map  : tuple of tuples of ints - represents a grid in the shape (nrows, ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
                
                Our "maze" the player is moving through.
        
        pos: tuple (int, int), optional
            Visualize our map to include a dot at this position.
        
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
        should_show = False
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111 ,aspect='equal')
            should_show = True
        
        curr_map = self

        if path: #If given a node, we've chosen a partial path.

            
            #row --> y axis, col --> x axis
            #Thus, to do (x,y), we need tuples of the form (c,r)
            path = [(c ,r) for r ,c in path][::-1]
            #Also reversing order of path
            
            #Convert pairs of elements (x,y) into two lists: X and Y
            X, Y = zip(*[ (x + 0.5, self.nrows - y - 0.5)   for x ,y in path])
            #Offset (+0.5, -0.5) is so that our dots are centered on each tile
            
            
            ###Plotting our path
            
            #Draw dotted line between each tile on path
            ax.plot(X, Y, 'o--',  markersize=4)
            
            #Color our starting point (X[-1],Y[-1]) as purple
            ax.plot(X[-1], Y[-1], 's', markersize=8, color='purple')

        
        #Convert string numbers into int numbers
        curr_map = curr_map[::-1]

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

        # add a title
        ax.set(title='Maze ' + self.name)
        
        if pos:
            (r,c)=pos
            
            ax.plot(c+.5, self.nrows-r-.5, 'bo', markersize=10)
        
        if should_show:
            plt.show()

###Used so we don't have to re-calculate the same tree multiple times
def memoize(function):
    """ ram cacher """
    memo = {}
    def wrapper(*args):
        if args not in memo:
            memo[tuple([*args])] = function(*args)
        return memo[tuple([*args])]
    return wrapper

#Use memoize to cache results of map2tree, save for re-use

#########################################################
#                                                       #
#                       DEFUNCT                         #
#                                                       #
#########################################################
@memoize
def maze2tree_defunct(maze):
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
                'pid': None,            #Parent ID: the root node has no parent
                'depth':0}            
            }           
    
    #BFS in progress
    # agenda = [(0, map_)] #(node, current map)
    agenda = deque([(0, maze)]) #(node, current map)
    
    while agenda: # in each loop, find and append children

        node, new_map = agenda.popleft() #Grab first element
        pos = tree[node]['pos'] #Current position
        
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
                      'depth': tree[node]['depth'] + 1, #Depth of our tree
                      'map': new_map.map} #New modified map

            new_node = max(tree ) +1 #Enumerate: max gives us most recent (highest index) node
            
            #Add our new node to the agenda: expand this path further
            
            agenda.append((new_node, new_map.update_map(path[-2], path[-1])))
            
            
            #Make the parent-child relation
            tree[node]['children'].add(new_node)
            #Add child to tree
            tree[new_node] = branch
            
    return tree

@memoize
def maze2graph(maze):
    """
    Converts our maze into a "state transition graph", representing the ways we can navigate through the maze.
    
    Every state is one maze, where n tiles are hidden.
    The particular tiles which are hidden, and the player's location, uniquely define the maze.
    
    The transition is given by which maze we choose to move to next.
    
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
    if type(maze) in (tuple, list): #If we have a grid, convert it.
        maze = grid2maze(maze)    

    ###Create tree to edit 
    root_maze = maze #A state is the position, and which tiles are hidden.
    
    graph = {root_maze: { 'children':{} } } #{'children':{childMaze: path}}
        
    #BFS in progress
    agenda = deque([ root_maze ]) #(node, current map)
    
    while agenda: # in each loop, find and append children
        parent_maze = agenda.popleft() #Grab first element
        
        #Try each new move
        for path in parent_maze.possible_paths(): 
            
            #Get all info about new condition
            new_pos = path[-1]
            child_maze = parent_maze.update_map(new_pos) #Create new maze 
            
            if not child_maze in graph: #New node!
                #branch = gen_branch(maze=child_maze) #Create new node
                
                
                graph[child_maze] = {'children':{}} #Add this node to the graph
                
                agenda.append(child_maze) #New nodes need to be explored further!
                
                # parent_maze.visualize()
                # child_maze.visualize()

            #Update parent's children
            children = graph[parent_maze]['children'] #Hoping this aliases correctly
            
            if not child_maze in children: #New child!
                children[child_maze] = path
            
            else: #Not new child: is this a better path?
                
                old_path = children[child_maze]
                
                if len(path) < len(old_path): #If this path is better, take it
                    
                    children[child_maze] = path #Update to new path!
        
    return graph


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

    
            