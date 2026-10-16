import { DenseMLP } from './ml/neural-network'
import { Cell } from './game-logic'

type Predictor = (input: Float32Array) => Float32Array;
type DenseMLPPredictor = DenseMLP["forwardPass"];

export interface BrainInterface {
    inputNumber: number,
    outputNumber: number,
    predictorInterface: Predictor
}

export class DenseMLPBrain implements BrainInterface {
    inputNumber: number;
    outputNumber: number;
    predictorInterface: DenseMLPPredictor;

    constructor(
        inputNumber: number,
        outputNumber: number,
        predictorInterface: DenseMLPPredictor,
    ) {
        this.inputNumber = inputNumber;
        this.outputNumber = outputNumber;
        this.predictorInterface = predictorInterface;
    }

    maskOutput(board: Array<Cell>): Float32Array {
        // we can just add this mask thanks to the property of 0 being neutral element 
        // of addition and -Infinity will crush everything elese!
        const boardSize: number | undefined = board.length;
        if (boardSize === undefined) {
            throw new Error("Mask is undefined!");
        }
        let maskedOutput: Float32Array = new Float32Array(boardSize);
        for (let boardIndex = 0; boardIndex < boardSize; boardIndex++) {
            const currentSymbol = board[boardIndex];
            if (currentSymbol == Cell.O || currentSymbol == Cell.X) {
                maskedOutput[boardIndex] = -Infinity;
            } else {
                maskedOutput[boardIndex] = 0;
            }
        }
        return maskedOutput;
    }

    chooseMove(input: Float32Array): number {
        return -1;
    }
}