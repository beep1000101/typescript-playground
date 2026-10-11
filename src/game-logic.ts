export type Cell = -1 | 0 | 1;

export function checkWinner(
    board: Cell[],
    lastMoveIndex: number,
    boardSize = 3,
    winLength = 3): number[] | null {
    const lastMoveRow = Math.floor(lastMoveIndex / boardSize);
    const lastMoveColumn = lastMoveIndex % boardSize;
    const playerSymbol = board[lastMoveIndex];

    if (playerSymbol === 0) {
        return null;
    }

    const directions: [number, number][] = [
        [0, 1],
        [1, 0],
        [1, 1],
        [1, -1]
    ];

    function scanDirection(
        rowStep: number,
        columnStep: number,
    ) {
        const matchingCells = [];

        let currentRow = lastMoveRow + rowStep;
        let currentColumn = lastMoveColumn + columnStep;

        while (
            currentRow >= 0 &&
            currentRow < boardSize &&
            currentColumn >= 0 &&
            currentColumn < boardSize &&
            board[currentRow * boardSize + currentColumn] === playerSymbol
        ) {
            matchingCells.push(currentRow * boardSize + currentColumn);

            currentRow += rowStep;
            currentColumn += columnStep;
        }

        return matchingCells;
    }

    for (const [rowStep, columnStep] of directions) {
        const forwardCells = scanDirection(rowStep, columnStep);
        const backwardCells = scanDirection(-rowStep, -columnStep);

        const winningLine = [
            ...backwardCells.reverse(),
            lastMoveIndex,
            ...forwardCells
        ];

        if (winningLine.length >= winLength) {
            return winningLine;
        }
    }

    return null;
}