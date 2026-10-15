import { DenseMLP } from './ml/neural-network'

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

    maskOutput(mask: Array<number>): Float32Array {
        return new Float32Array(this.outputNumber);
    }

    chooseMove(input: Float32Array): number {
        return -1;
    }
}