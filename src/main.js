import { checkWinner } from "./game-logic.js";

const board = new Int8Array(9);

let currentPlayer = "O";
let gameOver = false;
let movesPlayed = 0;

const gridElement = document.getElementById("grid-div");
const resetButton = document.getElementById("reset-button");
const statusElement = document.getElementById("status");

function resetBoard() {
    board.fill(0);

    const cells = gridElement.querySelectorAll(".cell");

    cells.forEach(cell => {
        cell.textContent = "";
        cell.classList.remove("winning");
    });

    currentPlayer = "O";
    gameOver = false;
    movesPlayed = 0;

    statusElement.textContent = "Player O's turn";
}

function setSymbol(element, index) {
    if (gameOver || board[index] !== 0) {
        return;
    }

    const symbol = currentPlayer === "X" ? 1 : -1;

    board[index] = symbol;
    element.textContent = currentPlayer;
    movesPlayed++;

    const winningCells = checkWinner(board, index);

    if (winningCells !== null) {
        gameOver = true;

        statusElement.textContent = `Player ${currentPlayer} wins!`;

        const cells = gridElement.querySelectorAll(".cell");

        winningCells.forEach(i => {
            cells[i].classList.add("winning");
        });

        return;
    }

    if (movesPlayed === 9) {
        gameOver = true;
        statusElement.textContent = "It's a draw!";
        return;
    }

    currentPlayer = currentPlayer === "X" ? "O" : "X";
    statusElement.textContent = `Player ${currentPlayer}'s turn`;
}

resetButton.addEventListener("click", resetBoard);

for (let i = 0; i < 9; i++) {
    const element = document.createElement("div");
    element.classList.add("cell");

    element.addEventListener("click", () => setSymbol(element, i));

    gridElement.appendChild(element);
}