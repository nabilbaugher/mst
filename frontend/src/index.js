import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import styled from 'styled-components';

import { createClient } from '@supabase/supabase-js';
import { v1 as uuid } from 'uuid';

// Supabase
const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
const supabaseKey = process.env.REACT_APP_SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

// const root = ReactDOM.createRoot(document.getElementById('root'));
// root.render(
//   <React.StrictMode>
//     <App />
//   </React.StrictMode>
// );

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

class GridSystem {
    constructor(matrices) {
        console.log('matrices');
        console.log(matrices);
        this.matrices = matrices;
        // We start with the first matrix in matrices
        this.matrix = JSON.parse(JSON.stringify(matrices[0]));
        this.setUpHelper();

        this.resetBoardNewMaze = this.resetBoardNewMaze.bind(this);
        this.setUpHelper = this.setUpHelper.bind(this);

        // Generate a random user ID for current user
        // Note: Currently, if one user refreshes the page, they will be considered a different user
        this.current_user_id = uuid();
    }

    resetBoardNewMaze() {
        // Reset the current matrix and make this.matrices = this.matrices[1:]
        // So if we take the first element in matrices matrices[0], it will give
        // us the next maze in our sequence
        console.log('this.matrices');
        console.log(this.matrices);
        this.matrix = JSON.parse(JSON.stringify(this.matrices[0]));
        this.matrices = this.matrices.slice(1, this.matrices.length);

        console.log('this.matrices');
        console.log(this.matrices);

        this.setUpHelper();
    }

    setUpHelper() {
        /*
        Helper function for constructor and resetBoardNewMaze,
        Sets up the player location and board
        */
        var playerX = null;
        var playerY = null;

        this.player_won = false;

        console.log('this.matrix in setUpHelper');
        console.log(this.matrix);

        for (let row = 0; row < this.matrix.length; row++) {
            for (let col = 0; col < this.matrix[row].length; col++) {
                if (this.matrix[row][col] == 5) {
                    playerX = col;
                    playerY = row;
                }
            }
        }
        console.log('coordinates: ' + playerX + ' ' + playerY);

        // Trying to keep track of original player location,
        // Might need it later for multiple mazes in a row
        this.original_playerX = playerX;
        this.original_playerY = playerY;

        this.uiContext = this.#getContext(420, 580, '#000');
        this.outlineContext = this.#getContext(0, 0, '#444');
        this.topContext = this.#getContext(0, 0, '#111', true);
        this.cellSize = 40;
        this.padding = 2;
        this.player = { x: playerX, y: playerY, color: 'orange' };
        // this.matrix[playerY][playerX] = 2;

        // Keep track of the keystroke sequence
        this.current_keystroke_sequence = [];

        document.addEventListener('keydown', this.#movePlayer);
    }

    // We need a function to reset

    async insertEntry({ tester_id, created_at, keystroke_sequences }) {
        /*
        Uploads new data entry to keystroke_sequence table in supabase
        */
        try {
            //setLoading(True)
            const update = {
                tester_id: tester_id,
                created_at,
                keystroke_sequences,
            };

            const { error } = await supabase
                .from('keystroke_sequence')
                .insert(update);

            if (error) {
                throw error;
            }
        } catch (error) {
            alert(error.message);
        } finally {
            console.log('Done');
            //setLoading(False)
        }
    }

    #isValidMove(x, y) {
        /* 
        Input: x: number of proposed steps to move right
                y: number of proposed steps to move down
        Output:
            true if the move is a valid move
            false otherwise
        */
        const new_square_val = this.matrix[this.player.y + y][this.player.x + x];
        const curr_square_val = this.matrix[this.player.y][this.player.x];
        console.log(curr_square_val);
        if (curr_square_val === 2) {
            // Already won
            return false;
        }
        if (
            new_square_val === 0 ||
            new_square_val === 6 ||
            new_square_val === 5 ||
            new_square_val === 2
        ) {
            if (new_square_val === 2) {
                this.player_won = true;
            }
            return true;
        }
        return false;
    }

    #updateMatrix(y, x, val) {
        this.matrix[y][x] = val;
    }

    #movePlayer = ({ keyCode }) => {
        /*
        Maps the arrow keys left, right, up and down to move actions
        */
        if (keyCode === 37) {
            const is_valid = this.#isValidMove(-1, 0);
            if (is_valid) {
                this.#updateMatrix(this.player.y, this.player.x, 6);
                if (this.player_won == false) {
                    this.#updateMatrix(this.player.y, this.player.x - 1, 8);
                }
                this.player.x--;
                this.current_keystroke_sequence.push('left');
                this.render();
            }
        } else if (keyCode === 39) {
            const is_valid = this.#isValidMove(1, 0);
            if (is_valid) {
                this.#updateMatrix(this.player.y, this.player.x, 6);
                if (this.player_won == false) {
                    this.#updateMatrix(this.player.y, this.player.x + 1, 8);
                }
                this.player.x++;
                this.current_keystroke_sequence.push('right');
                this.render();
            }
        } else if (keyCode === 38) {
            const is_valid = this.#isValidMove(0, -1);
            if (is_valid) {
                this.#updateMatrix(this.player.y, this.player.x, 6);
                if (this.player_won == false) {
                    this.#updateMatrix(this.player.y - 1, this.player.x, 8);
                }
                this.player.y--;
                this.current_keystroke_sequence.push('up');
                this.render();
            }
        } else if (keyCode === 40) {
            const is_valid = this.#isValidMove(0, 1);
            if (is_valid) {
                this.#updateMatrix(this.player.y, this.player.x, 6);
                if (this.player_won == false) {
                    this.#updateMatrix(this.player.y + 1, this.player.x, 8);
                }
                this.player.y++;
                this.current_keystroke_sequence.push('down');
                this.render();
            }
        }
    };

    #getCenter(w, h) {
        return {
            x: window.innerWidth / 2 - w / 2 + 'px',
            y: window.innerHeight / 2 - h / 2 + 'px',
        };
    }

    #getContext(w, h, color = '#111', isTransparent = false) {
        this.canvas = document.createElement('canvas');
        this.context = this.canvas.getContext('2d');
        this.width = this.canvas.width = w;
        this.height = this.canvas.height = h;
        this.canvas.style.position = 'absolute';
        this.canvas.style.background = color;
        if (isTransparent) {
            this.canvas.style.backgroundColor = 'transparent';
        }
        const center = this.#getCenter(w, h);
        this.canvas.style.marginLeft = center.x;
        this.canvas.style.marginTop = center.y;

        document.body.appendChild(this.canvas);

        return this.context;
    }

    #raycaster(map, pos) {
        // Retrieve the coordinates of the player's position
        let r = pos[0];
        let c = pos[1];

        // Retrieve the size of the map
        let nrows = map.length;
        let ncols = map[0].length;

        // Create a set to store the tiles currently visible to the player
        let observed = new Set();

        // 1st quadrant: +x, +y
        let nearestRightWall = ncols; // Can't see past the edge of the map

        // Limit to top half of the map: [r, r-1, r-2, ..., 0]
        for (let r_ = r; r_ >= 0; r_--) {
            // Retrieve the current row of the map
            let row = map[r_];

            // Wall stops our player from seeing past it
            let wall = Math.min(
                nearestRightWall, // Previous wall
                row.indexOf(3, c)
            ); // This row's closest wall: 3 is a wall

            // Both walls block: whichever wall is closer (min), will block more.

            // Limit to right half of the map
            let right = [];
            for (let c_ = c; c_ < wall; c_++) {
                right.push(c_); // All of the points we can see: between our position and the wall
            }
            observed.add(right.map((c_) => [r_, c_])); // Add the visible tiles to the set of observed tiles

            if (right.length == 0) {
                // If nothing new is seen, then walls are completely blocking our view.
                break;
            }

            nearestRightWall = right[right.length - 1] + 1; // Closest wall carries over as we move further away: still blocks
        }

        // 2nd quadrant: -x, +y
        // Getting left half of map by flipping map, and getting next "right" half
        let nearestLeftWall = ncols;
    }

    render() {
        /*
        Makes the visual representation of the maze
        */
        const w =
            (this.cellSize + this.padding) * this.matrix[0].length - this.padding;
        const h =
            (this.cellSize + this.padding) * this.matrix.length - this.padding;

        this.outlineContext.canvas.width = w;
        this.outlineContext.canvas.height = h;

        const center = this.#getCenter(w, h);
        this.outlineContext.canvas.style.marginLeft = center.x;
        this.outlineContext.canvas.style.marginTop = center.y;

        this.topContext.canvas.style.marginLeft = center.x;
        this.topContext.canvas.style.marginTop = center.y;

        for (let row = 0; row < this.matrix.length; row++) {
            for (let col = 0; col < this.matrix[row].length; col++) {
                const cellVal = this.matrix[row][col];
                const curr_val = this.matrix[row][col];

                // 8 is our player character, player color is moccasin
                const color_map = {
                    0: '#9c9c9c',
                    1: 'white',
                    2: '#d074a4',
                    3: '#b0943d',
                    4: 'white',
                    5: '#a1c38c',
                    6: 'white',
                    7: '#9c9c9c',
                    8: 'moccasin',
                };

                const color_assigned = color_map[curr_val];

                this.outlineContext.fillStyle = color_assigned;

                //this.outlineContext.fillStyle = color;
                this.outlineContext.fillRect(
                    col * (this.cellSize + this.padding),
                    row * (this.cellSize + this.padding),
                    this.cellSize,
                    this.cellSize
                );
            }
        }

        let btn = document.createElement('button');
        btn.innerHTML = 'Next trial';
        btn.style.position = 'absolute';
        btn.style.left = center.x;
        btn.style.top = '75%';
        btn.onclick = this.resetBoardNewMaze;

        // const Button = styled.button`
        // 	background-color: black;
        // 	color: white;
        // 	font-size: 20px;
        // 	padding: 10px 60px;
        // 	border-radius: 5px;
        // 	margin: 10px 0px;
        // 	cursor: pointer;
        // `;

        //<Button onClick={this.resetBoardNewMaze}>Disabled Button</Button>;

        // 	this.resetBoardNewMaze();
        // 	// if (this.player_won == false) {
        // 	// 	this.insertEntry({
        // 	// 		tester_id: this.current_user_id,
        // 	// 		created_at: String(Date.now()),
        // 	// 		keystroke_sequences: {
        // 	// 			1: this.current_keystroke_sequence
        // 	// 	}})
        // 	// }

        // 	// [TODO] Reset trial
        // };
        document.body.appendChild(btn);

        this.uiContext.font = '20px Courier';
        this.uiContext.fillStyle = 'white';

        if (this.player_won == true) {
            // Upload keystroke data
            this.current_keystroke_sequence.push('won');

            this.insertEntry({
                tester_id: this.current_user_id,
                created_at: String(Date.now()),
                keystroke_sequences: {
                    1: this.current_keystroke_sequence,
                },
            });

            this.uiContext.fillText('You win!!', 20, 30);
        }
        // else {
        // 	this.uiContext.fillText("Grid Based System", 20, 30);
        // }
    }
}

const gridMatrix1 = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 0, 3, 0, 3, 3],
    [3, 3, 3, 3, 0, 3, 0, 3, 3],
    [3, 5, 6, 6, 6, 6, 6, 6, 3],
    [3, 6, 3, 3, 3, 3, 3, 6, 2],
    [3, 6, 6, 6, 6, 6, 6, 6, 3],
    [3, 3, 0, 0, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
];

const gridMatrix2 = [
    [3, 6, 6, 6, 6, 6, 6, 3, 3],
    [3, 3, 3, 3, 0, 3, 0, 3, 3],
    [3, 3, 3, 3, 0, 3, 0, 3, 3],
    [3, 5, 6, 6, 6, 6, 6, 6, 3],
    [3, 6, 3, 3, 3, 3, 3, 6, 2],
    [3, 6, 6, 6, 6, 6, 6, 6, 3],
    [3, 3, 0, 0, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
];

// const maze_0 = [
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 6, 3, 3, 3]
// 	[3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 0, 0, 6, 6, 6, 0, 2, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 0, 3, 6, 3, 3, 0, 0, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 0, 3, 6, 3, 3, 3, 0, 3, 0, 3, 6, 3, 3, 3]
// 	[3, 0, 3, 6, 3, 0, 0, 0, 3, 0, 3, 6, 3, 3, 3]
// 	[3, 0, 0, 6, 3, 0, 0, 3, 3, 0, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 0, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// ];

// const maze_1 = [
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 0, 0, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 6, 6, 6, 0, 0, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 6, 3, 3, 0, 2, 0, 0, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 0, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 0, 0, 0, 3, 3, 0, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 0, 0, 3, 3, 0, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// ];

// const maze_2 = [
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 6, 6, 0, 0, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 0, 2, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 0, 0, 0, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 0, 0, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3]
// ];

// const maze_3 = [
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 6, 6, 0, 0, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 2, 0, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 0, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 0, 0, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 0, 0, 0, 0, 0, 0, 3, 6, 0, 0, 3]
// 	[3, 3, 3, 6, 0, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 0, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// ];

// const maze_4 = [
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 6, 3, 3, 3]
// 	[3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 0, 0, 6, 6, 6, 2, 0, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 0, 3, 6, 3, 3, 0, 0, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 0, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 0, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 0, 0, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3]
// ];

// const maze_5 = [
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 6, 3, 3, 3]
// 	[3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 0, 0, 6, 6, 6, 0, 0, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 0, 3, 6, 3, 3, 2, 0, 0, 0, 3, 6, 3, 3, 3]
// 	[3, 0, 3, 6, 3, 3, 3, 3, 3, 0, 3, 6, 3, 3, 3]
// 	[3, 0, 0, 6, 3, 3, 3, 3, 3, 0, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 0, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3]
// ];

// const maze_6 = [
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 6, 6, 6, 0, 2, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 6, 3, 3, 0, 0, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 0, 0, 0, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 0, 0, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// ];

// const maze_7 = [
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 6, 6, 0, 2, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 0, 0, 0, 0, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 0, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 0, 0, 6, 0, 0, 3]
// 	[3, 3, 3, 6, 3, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0]
// 	[3, 3, 3, 6, 3, 0, 3, 3, 3, 3, 3, 6, 3, 0, 0]
// 	[3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// ];

// const maze_8 = [
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 6, 6, 0, 0, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 0, 2, 3, 0, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 3, 0, 3, 3, 0, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 0, 0, 3, 3, 0, 0, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 0, 0, 0, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 3, 0, 0, 0, 3, 3, 3, 6, 3, 3, 3]
// 	[3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// ];

// const maze_9 = [
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 6, 6, 6, 0, 0, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 6, 3, 3, 0, 2, 0, 0, 3, 6, 3, 0, 3]
// 	[3, 3, 3, 6, 3, 3, 3, 3, 3, 0, 3, 6, 3, 0, 3]
// 	[3, 0, 0, 6, 3, 3, 3, 3, 3, 0, 0, 6, 3, 3, 3]
// 	[3, 0, 0, 6, 3, 3, 3, 3, 3, 0, 0, 6, 0, 0, 3]
// 	[3, 0, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3]
// 	[3, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 0, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// 	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
// ];

const gridSystem = new GridSystem([gridMatrix1, gridMatrix2]);
gridSystem.render();
