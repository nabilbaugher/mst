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

        this.current_trial_number = 1;

        this.resetBoardNewMaze = this.resetBoardNewMaze.bind(this);
        this.setUpHelper = this.setUpHelper.bind(this);
        this.submitWorkerID = this.submitWorkerID.bind(this);
        this.showAfterEndScreen = this.showAfterEndScreen.bind(this);
        this.nextTrialClick = this.nextTrialClick.bind(this);
        // this.showEndScreen = this.showEndScreen.bind(this);
        // this.render = this.render.bind(this);
        // this.render = this.render.bind(this);

        // Generate a random user ID for current user
        // Note: Currently, if one user refreshes the page, they will be considered a different user
        this.current_user_id = uuid();

        document.addEventListener('keydown', this.#movePlayer);

        this.setUpHelper();

    }

    resetBoardNewMaze() {
        /* Reset the current matrix and make this.matrices = this.matrices[1:]
         So if we take the first element in matrices matrices[0], it will give
         us the next maze in our sequence */
        console.log('this.matrices');
        console.log(this.matrices);

        var you_win_text = document.getElementById("you_win_text");
        you_win_text.innerHTML = "";

        this.matrices = this.matrices.slice(1, this.matrices.length);
        this.current_trial_number += 1;

        if (this.matrices.length == 0) {
            // If there are no more matrices in our array, 
            // Display a text field for the user to enter worker ID
            this.showEndScreen();

        } else {
            this.matrix = JSON.parse(JSON.stringify(this.matrices[0]));
            console.log('this.matrices');
            console.log(this.matrices);


            this.setUpHelper();
        }

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
        //this.uiContext = this.#getContext(720, 580, '#000');
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

    async insertEntryTrial({ tester_id, keystroke_sequence }) {
        /*
        Uploads new data entry to trials table in supabase
        */
        try {
            //setLoading(True)
            const update = {
                entry_id: uuid(),
                tester_id: tester_id,
                created_at: String(Date.now()),
                keystroke_sequence,
            };

            const { error } = await supabase
                .from('trials')
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

    async insertEntryWorkerID({ tester_id, worker_id }) {
        /*
        Uploads new data entry to worker_id table in supabase
        */
        try {
            //setLoading(True)
            const update = {
                entry_id: uuid(),
                tester_id: tester_id,
                created_at: String(Date.now()),
                worker_id
            };

            const { error } = await supabase
                .from('worker_ids')
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
                    if ((observations[k][0] === i && observations[k][1] === j)
                        && !(observations[k][0] === this.player.y && observations[k][1] === this.player.x)
                        && this.matrix[i][j] !== 2) {
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

    showEndScreen() {
        // Clear screen
        document.body.innerHTML = "";

        // Show text
        const para = document.createElement("p");
        const thank_you_string = "Thank you for completing this survey! Please enter your Worker ID below to ensure compensation for this task. You can find this on your MTurk Dashboard or in the upper left corner of the Worker website.";
        para.textContent = thank_you_string;
        para.style.position = 'absolute';
        para.style.left = "10%";
        para.style.top = "10%";
        document.body.appendChild(para);

        // Input for person to input their MTurk worker id
        var worker_id_input = document.createElement("input");
        worker_id_input.type = "text";
        worker_id_input.placeholder = "Worker ID";
        worker_id_input.className = "worker_id_input";
        worker_id_input.id = "worker_id_input";
        worker_id_input.style.position = 'absolute';
        worker_id_input.style.left = "10%";
        worker_id_input.style.top = "20%";
        document.body.appendChild(worker_id_input);

        // Button for person to submit
        let btn = document.createElement('button');
        btn.innerHTML = 'Submit';
        btn.style.position = 'absolute';
        btn.style.left = "10%";
        btn.style.top = '30%';
        document.body.appendChild(btn);



        btn.addEventListener("click", this.submitWorkerID);
    }

    showAfterEndScreen() {
        // Clear screen
        document.body.innerHTML = "";

        // Show text
        const para = document.createElement("p");
        const thank_you_string = "Thank you!"
        para.textContent = thank_you_string;
        para.style.position = 'absolute';
        para.style.left = "10%";
        para.style.top = "10%";
        document.body.appendChild(para);
    }

    submitWorkerID() {

        var worker_id = document.getElementById("worker_id_input").value;
        console.log("worker_id");
        console.log(worker_id);
        this.insertEntryWorkerID({
            tester_id: this.current_user_id,
            // created_at: String(Date.now()),
            worker_id: worker_id

        });

        this.showAfterEndScreen();
    }

    showEndScreen() {
        // Clear screen
        document.body.innerHTML = "";

        // Show text
        const para = document.createElement("p");
        const thank_you_string = "Thank you for completing this survey! Please enter your Worker ID below to ensure compensation for this task. You can find this on your MTurk Dashboard or in the upper left corner of the Worker website.";
        para.textContent = thank_you_string;
        para.style.position = 'absolute';
        para.style.left = "10%";
        para.style.top = "10%";
        document.body.appendChild(para);

        // Input for person to input their MTurk worker id
        var worker_id_input = document.createElement("input");
        worker_id_input.type = "text";
        worker_id_input.placeholder = "Worker ID";
        worker_id_input.className = "worker_id_input";
        worker_id_input.id = "worker_id_input";
        worker_id_input.style.position = 'absolute';
        worker_id_input.style.left = "10%";
        worker_id_input.style.top = "20%";
        document.body.appendChild(worker_id_input);

        // Button for person to submit
        let btn = document.createElement('button');
        btn.innerHTML = 'Submit';
        btn.style.position = 'absolute';
        btn.style.left = "10%";
        btn.style.top = '30%';
        document.body.appendChild(btn);



        btn.addEventListener("click", this.submitWorkerID);
    }

    showAfterEndScreen() {
        // Clear screen
        document.body.innerHTML = "";

        // Show text
        const para = document.createElement("p");
        const thank_you_string = "Thank you!"
        para.textContent = thank_you_string;
        para.style.position = 'absolute';
        para.style.left = "10%";
        para.style.top = "10%";
        document.body.appendChild(para);
    }

    submitWorkerID() {

        var worker_id = document.getElementById("worker_id_input").value;
        console.log("worker_id");
        console.log(worker_id);
        this.insertEntryWorkerID({
            tester_id: this.current_user_id,
            // created_at: String(Date.now()),
            worker_id: worker_id

        });

        this.showAfterEndScreen();
    }

    nextTrialClick() {
        if (this.player_won == true) {
            this.player_won == false;
            this.resetBoardNewMaze();
        }
        else {
            var you_win_text = document.getElementById("you_win_text");
            you_win_text.innerHTML = "Error: Please complete this <br/> maze before continuing"
        }
    }

    render() {
        /*
        Makes the visual representation of the maze
        */

        // Remove old instruction string
        const element = document.getElementById('instruction_string');
        if (element) {
            element.remove();
        }


        const para = document.createElement("p");
        //const instruction_string =  git
        para.innerHTML = "Thank you for participating in our survey! You will be presented with 10 different types of mazes. Use the arrow keys → ← ↑ ↓ to move until you find the exit and win! The exit will be in a gray square. (Try to avoid refreshing the page, mazes will reset) </br>";
        para.innerHTML += "Current Trial: " + String(this.current_trial_number);
        //para.textContent = instruction_string;
        para.id = "instruction_string";
        para.style.position = 'absolute';
        para.style.left = "10%";
        para.style.top = "5%";
        document.body.appendChild(para);

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
        btn.style.top = '85%';
        btn.addEventListener("click", this.nextTrialClick);


        // You win text

        var you_win_text = document.getElementById("you_win_text");

        if (you_win_text == null) {
            you_win_text = document.createElement("h2");
        }

        //const thank_you_string = "";
        you_win_text.id = "you_win_text";
        you_win_text.innerHTML = "";
        // you_win_text.textContent = "you win"
        you_win_text.style.position = 'absolute';
        you_win_text.style.left = "10%";
        you_win_text.style.top = "50%";
        document.body.appendChild(you_win_text);
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

            var trial_dict = {}
            trial_dict[this.current_trial_number] = this.current_keystroke_sequence;

            this.insertEntryTrial({
                tester_id: this.current_user_id,
                // created_at: String(Date.now()),
                keystroke_sequence: trial_dict,
            });

            //var win_text = document.getElementById("you_win_text");
            you_win_text.innerHTML += "You win!<br />";
            you_win_text.innerHTML += "Click Next Trial to continue<br />";
            //this.uiContext.fillText('You win!!', 20, 30);
        }
        // else {
        // 	this.uiContext.fillText("Grid Based System", 20, 30);
        // }
    }
}

// const testMatrix1 = [
//     [3, 3, 3, 3, 3, 3, 3, 3, 3],
//     [3, 3, 3, 3, 0, 3, 0, 3, 3],
//     [3, 3, 3, 3, 0, 3, 0, 3, 3],
//     [3, 5, 6, 6, 6, 6, 6, 6, 3],
//     [3, 6, 3, 3, 3, 3, 3, 6, 2],
//     [3, 6, 6, 6, 6, 6, 6, 6, 3],
//     [3, 3, 0, 0, 3, 3, 3, 3, 3],
//     [3, 3, 3, 3, 3, 3, 3, 3, 3],
// ];

// const testMatrix2 = [
//     [3, 6, 6, 6, 6, 6, 6, 3, 3],
//     [3, 3, 3, 3, 0, 3, 0, 3, 3],
//     [3, 6, 6, 3, 0, 3, 0, 3, 3],
//     [3, 5, 6, 6, 6, 6, 6, 6, 3],
//     [3, 6, 3, 3, 3, 3, 3, 6, 2],
//     [3, 6, 6, 6, 6, 6, 6, 6, 3],
//     [3, 3, 0, 0, 3, 3, 3, 3, 3],
//     [3, 3, 3, 3, 3, 3, 3, 3, 3],
// ];

// const testMatrix3 = [
//     [3, 6, 6, 6, 6, 6, 6, 3, 3],
//     [3, 6, 6, 6, 6, 6, 6, 6, 3],
//     [3, 6, 6, 3, 0, 3, 0, 3, 3],
//     [3, 5, 6, 6, 6, 6, 6, 6, 3],
//     [3, 6, 3, 3, 3, 3, 3, 6, 2],
//     [3, 6, 6, 6, 6, 6, 6, 6, 3],
//     [3, 3, 0, 0, 3, 3, 3, 3, 3],
//     [3, 3, 3, 3, 3, 3, 3, 3, 3],
// ];



const maze_0 = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 6, 3, 3, 3],
    [3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 0, 0, 6, 6, 6, 0, 2, 3, 3, 3, 6, 3, 3, 3],
    [3, 0, 3, 6, 3, 3, 0, 0, 3, 3, 3, 6, 3, 3, 3],
    [3, 0, 3, 6, 3, 3, 3, 0, 3, 0, 3, 6, 3, 3, 3],
    [3, 0, 3, 6, 3, 0, 0, 0, 3, 0, 3, 6, 3, 3, 3],
    [3, 0, 0, 6, 3, 0, 0, 3, 3, 0, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 3, 3, 3, 3, 3, 0, 0, 6, 3, 3, 3],
    [3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
];

const maze_1 = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3],
    [3, 3, 3, 6, 6, 6, 0, 0, 3, 3, 3, 6, 3, 0, 3],
    [3, 3, 3, 6, 3, 3, 0, 2, 0, 0, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 3, 3, 3, 3, 3, 0, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 0, 0, 0, 3, 3, 0, 0, 6, 3, 3, 3],
    [3, 3, 3, 6, 3, 0, 0, 3, 3, 0, 0, 6, 3, 3, 3],
    [3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
];

const maze_2 = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 6, 3, 3, 3],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 6, 6, 0, 0, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 3, 3, 0, 2, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 0, 0, 0, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 3, 0, 0, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3],
];

const maze_3 = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 3, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 6, 6, 0, 0, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 3, 3, 2, 0, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 3, 3, 0, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 3, 0, 0, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 0, 0, 0, 0, 0, 0, 3, 6, 0, 0, 3],
    [3, 3, 3, 6, 0, 3, 3, 3, 3, 3, 3, 6, 3, 0, 3],
    [3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
];

const maze_4 = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 6, 3, 3, 3],
    [3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 0, 0, 6, 6, 6, 2, 0, 3, 3, 3, 6, 3, 3, 3],
    [3, 0, 3, 6, 3, 3, 0, 0, 3, 3, 3, 6, 3, 3, 3],
    [3, 0, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 0, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 0, 0, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3],
    [3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3],
    [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3],
    [3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3],
];

const gridSystem = new GridSystem([maze_0, maze_1, maze_2, maze_3, maze_4]);
gridSystem.render();
