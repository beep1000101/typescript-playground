export function checkWinner(board, lastMoveIndex, boardSize = 3, winLength = 3) {
    const lastMoveRow = Math.floor(lastMoveIndex / boardSize);
    const lastMoveColumn = lastMoveIndex % boardSize;
    const playerSymbol = board[lastMoveIndex];

    if (playerSymbol === 0) {
        return null;
    }

    const directions = [
        [0, 1],
        [1, 0],
        [1, 1],
        [1, -1]
    ];

    function scanDirection(rowStep, columnStep) {
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