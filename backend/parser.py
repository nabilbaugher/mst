import csv
from maze import Maze, maze2tree, grid2maze

#Data pulled as csv


directions = {'up':(0 ,1),
 'down':(0 ,-1),
 'left':(-1 ,0),
 'right':(1 ,0)}
#{maze_id: maze }
mazes = {}

def get_csv(filename):
    """ Turn CSV into a list of dictionaries. """ 
    
    out = []
    with open(f'{filename}.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            out.append(row)
            
    return out



def sample_convert(datapoint):
    """ Turn a dictionary into a ??? """
    
    subject = datapoint["tester_id"]
    maze_name = 'unknown' #Where is the maze identifier?
    keystroke_sequence = datapoint["keystroke_sequence"]

def parse_keystrokes(maze, keystroke_sequence):
    """ Convert keystrokes into our path through the maze, and the nodes we move through. """
    
    tree = maze2tree(maze) #Get node
    
    path=   [maze.start]
    nodes = [0]
    
    for stroke in keystroke_sequence: #Each action the user takes
        
        #Last position
        pos =  path[-1]
        node = nodes[-1]
        
        
        #How did our player moved, based on this keystroke?
        shift = directions[stroke] 
        new_pos = maze.move(pos, shift)
        
        if new_pos==pos: #Nothing changed
            continue
        
        path.append(new_pos) #Otherwise, add to path
        
        
        #Check if we have moved into a new node
        children = tree[node]['children'] #Possible future nodes
        
        for child in children: 
            node_pos = child['pos']
            
            if node_pos == new_pos: #We have a match!
                nodes.append(node)
    
    
    #Finished moving through maze
    return {'nodes': nodes, 'path': path}

if __name__ == "__main__":
    file = get_csv('trials_rows')