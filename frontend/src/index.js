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
        // console.log('matrices');
        // console.log(matrices);
        this.matrices = matrices;
        // We start with the first matrix in matrices
        this.matrix = JSON.parse(JSON.stringify(matrices[0]));

        this.resetBoardNewMaze = this.resetBoardNewMaze.bind(this);
        this.setUpHelper = this.setUpHelper.bind(this);
        // this.render = this.render.bind(this);

        // Generate a random user ID for current user
        // Note: Currently, if one user refreshes the page, they will be considered a different user
        this.current_user_id = uuid();

        this.setUpHelper();
    }

    resetBoardNewMaze() {
        // Reset the current matrix and make this.matrices = this.matrices[1:]
        // So if we take the first element in matrices matrices[0], it will give
        // us the next maze in our sequence
        // console.log('this.matrices');
        // console.log(this.matrices);

        this.matrices = this.matrices.slice(1, this.matrices.length);
        this.matrix = JSON.parse(JSON.stringify(this.matrices[0]));

        // console.log('this.matrices');
        // console.log(this.matrices);


        //this.setUpHelper = this.setUpHelper.bind(this);

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

        for (let row = 0; row < this.matrix.length; row++) {
            for (let col = 0; col < this.matrix[row].length; col++) {
                if (this.matrix[row][col] === 5) {
                    playerX = col;
                    playerY = row;
                }
            }
        }

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

        this.render();
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
                if (this.player_won === false) {
                    this.#updateMatrix(this.player.y, this.player.x - 1, 8);
                }
                this.player.x--;
                this.current_keystroke_sequence.push('left');
            }
        } else if (keyCode === 39) {
            const is_valid = this.#isValidMove(1, 0);
            if (is_valid) {
                this.#updateMatrix(this.player.y, this.player.x, 6);
                if (this.player_won === false) {
                    this.#updateMatrix(this.player.y, this.player.x + 1, 8);
                }
                this.player.x++;
                this.current_keystroke_sequence.push('right');
            }
        } else if (keyCode === 38) {
            const is_valid = this.#isValidMove(0, -1);
            if (is_valid) {
                this.#updateMatrix(this.player.y, this.player.x, 6);
                if (this.player_won === false) {
                    this.#updateMatrix(this.player.y - 1, this.player.x, 8);
                }
                this.player.y--;
                this.current_keystroke_sequence.push('up');
            }
        } else if (keyCode === 40) {
            const is_valid = this.#isValidMove(0, 1);
            if (is_valid) {
                this.#updateMatrix(this.player.y, this.player.x, 6);
                if (this.player_won === false) {
                    this.#updateMatrix(this.player.y + 1, this.player.x, 8);
                }
                this.player.y++;
                this.current_keystroke_sequence.push('down');
            }
        }
        this.#update_map();
        this.render();
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

    #update_map() {
        let observations = this.#newObservations(); //Get observations
        //Update map to reflect observations
        for (let i = 0; i < this.matrix.length; i++) {
            for (let j = 0; j < this.matrix[i].length; j++) {
                for (let k = 0; k < observations.length; k++) {
                    if ((observations[k][0] === i && observations[k][1] === j) && !(observations[k][0] === this.player.y && observations[k][1] === this.player.x)) {
                        this.matrix[i][j] = 6;
                    }
                }
            }
        }
    }

    #newObservations() {
        let observed = this.#raycaster(this.matrix, [this.player.y, this.player.x]);
        var newObservations = []

        for (var i = 0; i < observed.length; i++) {
            let r = observed[i][0]
            let c = observed[i][1]
            if (this.matrix[r][c] !== 0 && this.matrix[r][c] !== 2) {
                continue;
            }
            newObservations.push([r, c]);
        }
        return newObservations;
    }

    #index_from_row_col(row, col, ncols) {
        return row * ncols + col
    }

    #row_col_from_index(index, ncols) {
        return [Math.floor(index / ncols), index % ncols]
    }

    #get_index_of(arr, val, start = 0) {
        for (let i = start; i < arr.length; i++) {
            if (arr[i] === val) {
                return i
            }
        }
        return -1
    }

    #raycaster(matrix, position) {
        // returns a list of all cells visible from the given position
        // a cell is visible if each of it's four corners is visible from the corresponding corner of the position
        // a corner is visible if it is not blocked by a wall

        const r = position[0]
        const c = position[1]
        const observed = new Set()
        const nrows = matrix.length
        const ncols = matrix[0].length

        // first quadrant
        // limit to top half of the maze
        let nearest_right_wall = ncols

        for (let r_ = r; r_ >= 0; r_--) {
            const row_ = [...matrix[r_]]
            const wall = Math.min(nearest_right_wall, this.#get_index_of(row_, 3, c))
            const right = []
            for (let c_ = c; c_ < wall; c_++) {
                right.push(c_)
            }
            if (right.length > 0) {
                right.forEach(item => observed.add(this.#index_from_row_col(r_, item, ncols)))
            } else {
                break
            }
            nearest_right_wall = right[right.length - 1] + 1
        }


        // second quadrant
        // limit to left half of the maze
        let nearest_left_wall = ncols

        for (let r_ = r; r_ >= 0; r_--) {
            const row_ = [...matrix[r_]].reverse()
            const flipped_c = ncols - c
            const wall = Math.min(nearest_left_wall, this.#get_index_of(row_, 3, flipped_c - 1))
            const left = []
            for (let c_ = flipped_c; c_ < wall; c_++) {
                left.push(c_)
            }
            if (left.length > 0) {
                left.forEach(item => observed.add(this.#index_from_row_col(r_, ncols - item - 1, ncols)))
            } else {
                break
            }
            nearest_left_wall = left[left.length - 1] + 1
        }

        // third quadrant
        nearest_left_wall = ncols

        for (let r_ = r; r_ < nrows; r_++) {
            const row_ = [...matrix[r_]].reverse()
            const flipped_c = ncols - c
            const wall = Math.min(nearest_left_wall, this.#get_index_of(row_, 3, flipped_c - 1))
            const left = []
            for (let c_ = flipped_c; c_ < wall; c_++) {
                left.push(c_)
            }
            if (left.length > 0) {
                left.forEach(item => observed.add(this.#index_from_row_col(r_, ncols - item - 1, ncols)))
            } else {
                break
            }
            nearest_left_wall = left[left.length - 1] + 1
        }

        // fourth quadrant
        nearest_right_wall = ncols

        for (let r_ = r; r_ < nrows; r_++) {
            const row_ = [...matrix[r_]]
            const wall = Math.min(nearest_right_wall, this.#get_index_of(row_, 3, c))
            const right = []
            for (let c_ = c; c_ < wall; c_++) {
                right.push(c_)
            }
            if (right.length > 0) {
                right.forEach(item => observed.add(this.#index_from_row_col(r_, item, ncols)))
            } else {
                break
            }
            nearest_right_wall = right[right.length - 1] + 1
        }

        let result = []
        observed.forEach(item => result.push(this.#row_col_from_index(item, ncols)))
        return result
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
        btn.addEventListener("click", this.resetBoardNewMaze);
        // btn.onclick = function() {this.resetBoardNewMaze()};

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
        // 	// if (this.player_won === false) {
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

        if (this.player_won === true) {
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
    [3, 0, 0, 0, 6, 0, 0, 0, 3],
    [3, 0, 0, 3, 6, 3, 0, 0, 3],
    [3, 0, 0, 3, 5, 6, 6, 6, 3],
    [3, 0, 0, 3, 3, 3, 0, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
];

const gridMatrix2 = [
    [3, 6, 6, 6, 6, 6, 6, 3, 3],
    [3, 3, 3, 3, 0, 3, 0, 3, 3],
    [3, 6, 6, 3, 0, 3, 0, 3, 3],
    [3, 5, 6, 6, 6, 6, 6, 6, 3],
    [3, 6, 3, 3, 3, 3, 3, 6, 2],
    [3, 6, 6, 6, 6, 6, 6, 6, 3],
    [3, 3, 0, 0, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
];

const gridMatrix3 = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 0, 0, 0, 6, 0, 0, 0, 3],
    [3, 0, 0, 3, 6, 3, 0, 0, 3],
    [3, 0, 0, 3, 5, 3, 0, 0, 3],
    [3, 0, 0, 3, 3, 3, 0, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
]
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

const gridSystem = new GridSystem([gridMatrix1, gridMatrix2, gridMatrix3]);
gridSystem.render();
