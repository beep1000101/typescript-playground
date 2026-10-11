import { checkWinner, type Cell } from "./game-logic.js";

enum CurrentPlayer {
    O = "O",
    X = "X",
}

const board: Cell[] = Array<Cell>(9).fill(0);

let currentPlayer: CurrentPlayer = CurrentPlayer.O;
let gameOver: boolean = false;
let movesPlayed: number = 0;

function getRequiredElement<T extends Element>(selector: string): T {
    const element = document.querySelector<T>(selector);
    if (element === null) {
        throw new Error(`Missing required element: ${selector}`);
    }
    return element;
}

const gridElement = getRequiredElement<HTMLDivElement>("#grid-div");
const resetButton = getRequiredElement<HTMLButtonElement>("#reset-button");
const statusElement = getRequiredElement<HTMLDivElement>("#status");

function resetBoard(): void {
    board.fill(0);

    const cells = gridElement.querySelectorAll<HTMLDivElement>(".cell");

    cells.forEach(cell => {
        cell.textContent = "";
        cell.classList.remove("winning");
    });

    currentPlayer = CurrentPlayer.O;
    gameOver = false;
    movesPlayed = 0;

    statusElement.textContent = "Player O's turn";
}

function setSymbol(element: HTMLDivElement, index: number) {
    if (gameOver || board[index] !== 0) {
        return;
    }

    const symbol = currentPlayer === CurrentPlayer.X ? 1 : -1;

    board[index] = symbol;
    element.textContent = currentPlayer;
    movesPlayed++;

    const winningCells = checkWinner(board, index);

    if (winningCells !== null) {
        gameOver = true;

        statusElement.textContent = `Player ${currentPlayer} wins!`;

        const cells = gridElement.querySelectorAll<HTMLDivElement>(".cell");

        winningCells.forEach(index => {
            const cell = cells.item(index);
            if (cell !== null) {
                cell.classList.add("winning");
            }
        });

        return;
    }

    if (movesPlayed === 9) {
        gameOver = true;
        statusElement.textContent = "It's a draw!";
        return;
    }

    currentPlayer = currentPlayer === CurrentPlayer.X ? CurrentPlayer.O : CurrentPlayer.X;
    statusElement.textContent = `Player ${currentPlayer}'s turn`;
}

resetButton.addEventListener("click", resetBoard);

for (let i = 0; i < 9; i++) {
    const element = document.createElement("div");
    element.classList.add("cell");

    element.addEventListener("click", () => setSymbol(element, i));

    gridElement.appendChild(element);
}