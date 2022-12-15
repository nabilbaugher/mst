import csv
from maze import Maze, maze2graph, grid2maze
from test_mazes import mazes, graphs#, maze2graph_dict
#Data pulled as csv

"""
This file is used to convert our experimental data into a form we can compute with.

========================================================

A short summary of the various functions in this file:
    
    get_csv(filename): return list
        Read csv file, turn it into python data.
        
    parse_keystrokes(maze, keystrokes): return dict
        Reads the raw keystrokes sent out by our experimenter, and uses them to determine where our player 
        moved.
        
    convert_subject(datapoint): return dict
        Takes in data for one subject, and converts it to a format our software can interpret.
        
    convert_data(data): return dict
        Takes in ALL data, and converts it into a format our software can interpret.
        
    decisions_to_subject_decisions(decisions): return subject_decisions
        Convert decisions of various subjects into a more useful format.
"""



directions = {'up':  (-1, 0),
             'down': ( 1, 0),
             'left': ( 0,-1),
             'right':( 0, 1)}



def get_csv(filename):
    """ Turn CSV into a list of dictionaries. """ 
    
    out = []
    with open(f'{filename}.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            out.append(row)
            
    return out


def parse_keystrokes(maze, graph, keystrokes):
    """
    Convert keystrokes into our path through the maze, and the nodes we move through. 

    Parameters
    ----------
    maze : Maze object, our maze to navigate. Stores a maze.map object:
        
        map: tuple of tuples of ints - represents a grid in the shape (maze.nrows, maze.ncols)
               -tuple of tuples    has length = nrows, 
               -each tuple of ints has length = ncols.
               
               Our "maze" the player is moving through.
               
    keystrokes : list of str
        The sequence of keys our player pressed to navigate the maze. Relevant strings are 
        'up', 'down', 'left', 'right', 'win'.

    Returns
    -------
    out: dictionary with str keys, and list vals
        nodes: The nodes our player moved through in the maze tree. Each elem of list is an int: a node id.
        
        path: The tiles our player moved through on the maze. Each elem is a tuple representing a position.

    """
    
    
    path=   [maze.start]
    
    nodes = [maze]  #First node => most hidden squares!
    
    node_changes = []
    
    for stroke in keystrokes: #Each action the user takes
        
        if stroke=='won': #We're done
            break
    
        #Previous position
        curr_pos =  path[-1]
        curr_node = nodes[-1]
        
        curr_maze = curr_node
        
        #How did our player move, based on this keystroke?
        shift = directions[stroke] 
        new_pos = maze.move(curr_pos, shift)
        
        if new_pos==curr_pos: #Nothing changed
            continue
        
        path.append(new_pos) #Otherwise, add to path
        
        new_maze = curr_maze.update_map(new_pos)
        
        #Check if we've revealed anything - the map would have changed
        if curr_maze.map != new_maze.map:
            
            nodes.append(new_maze) #We've move into a new node!
            node_changes.append(new_pos)

    
    
    #Finished moving through maze
    return {'nodes': nodes, 'path': path, 'node_changes': node_changes}


def convert_subject(datapoint):
    """
    Interprets data for one subject in one play session. Converts it into the format we need: 
    converts keystrokes into a path through the map, along with the corresponding nodes we pass through.
    
    
    Parameters
    ----------
    datapoint : dictionary
        Represents all of the actions taken by our one subject across several maps.
        
        Contains 'keystroke_sequence' key:
            keystrokes : list of str
                The sequence of keys our player pressed to navigate the maze. Relevant strings are 
                'up', 'down', 'left', 'right', 'win'.


    Returns
    -------
    output : Dictionary of dictionaries.
    
        Outer dictionary separates different mazes from each other.
        
        Inner dictionary contains the path the user took through the maze, and the nodes they passed through
        in our tree representation.

    """
    
    #This dictionary in our data is stored as a string....
    mazes_and_keystrokes = eval ( datapoint["keystroke_sequence"] )
    
    output = {} #{maze: {path: [], nodes: []}  }
    
    #Each maze has its own path
    for maze_name, keystrokes in mazes_and_keystrokes.items():
        
        maze = mazes[maze_name] #Get actual maze
        graph = graphs[maze_name] #Get matching tree
        
        info = parse_keystrokes(maze, graph, keystrokes)
        
        output[maze_name] = info #Save the data 
        
    return output


def convert_data(data):
    """
    Take all of our data, and convert it into the format we need. First, separated by subject. 
    Then, converts keystrokes into a path through the map, along with the corresponding nodes we pass through.

    Parameters
    ----------
    data : list of dictionaries 
        Each list is a datapoint.
    
        datapoint : dictionary
            Represents all of the actions taken by our one subject across several maps.
            
            Contains 'keystroke_sequence':
                keystrokes : list of str
                    The sequence of keys our player pressed to navigate the maze. Relevant strings are 
                    'up', 'down', 'left', 'right', 'win'.

    Returns
    -------
    decisions : dictionary of dictionaries of dictionaries
        outer dictionary: Dictionary of test subjects
            
        middle dictionary: Dictionary of mazes
            
        inner dictionary: Dictionary containing how our player moved: either nodes, paths, or node_changes
        
            Stores the decisions made by every single subject we've tested.

    """
    
    decisions = {}
    
    for datapoint in data:
        subject = datapoint["tester_id"]
        
        new_subject_decisions = convert_subject(datapoint) #Convert the data for a single subject.
        
        decisions.setdefault(subject, {}).update(new_subject_decisions)
        
    return decisions


def decisions_to_subject_decisions(decisions):
    """
    Convert all our player decisions into a more useful format.

    Parameters
    ----------
    decisions : dictionary of dictionaries of dictionaries
        outer dictionary: Dictionary of test subjects
            
        middle dictionary: Dictionary of mazes
            
        inner dictionary: Dictionary containing how our player moved: either nodes, or paths
        
            Stores the decisions made by every single subject we've tested.

    Returns
    -------
    subject_decisions: dictionary where key is subject, value is a decisions_list
        decisions_list : list of tuples (Maze, Maze)
            Each tuple represents one decision our player made, moving between two nodes: (parent_maze, child_maze)
            To reach this node, our player has to have gone from its parent node, and chosen this option.
            
            This list in total represents every decision our player made.

    """
    
    subject_decisions = {} #New format: list of all of our decisions for each subject
    
    for subject, subject_data in decisions.items():
        subject_decisions[subject] = [] #Default
        
        for maze, maze_data in subject_data.items():
            nodes = maze_data['nodes'] #Get all of the nodes for this pair
            
            #Each decision is an event: making one choice means moving between two nodes
            new_decisions = [( nodes[i],nodes[i+1] ) 
                             for i in range(len(nodes)-1) ]
            
            subject_decisions[subject].extend( new_decisions )  #All of the decisions for one map
            
    return subject_decisions



if __name__ == "__main__":
    file = get_csv('trials_rows')
    
    u,d,l,r = 'up', 'down', 'left', 'right'
    
    # keystrokes_test = [u,u,l,l,d]
    # tree = maze2tree(Maze2)
    
    # result = parse_keystrokes(Maze2, keystrokes_test) #Go right to exit
    
    decisions = convert_data(file)
    
    subject_decisions = decisions_to_subject_decisions(decisions)
    
    Maze1 = mazes['1']
    
    m = decisions['02a063a0-768f-11ed-9574-33a31f13e24c']
    
    n = m['1']
    
    ex_decisions = subject_decisions['02a063a0-768f-11ed-9574-33a31f13e24c']
    
    
    maze = Maze1
    
    for s in n['path']:
        maze = maze.update_map(pos=s)
        print(s)
        if s in n['node_changes']:
            print("Bazinga uwu") #New node!
        
        maze.visualize(s)