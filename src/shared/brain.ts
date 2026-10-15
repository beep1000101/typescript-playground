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
}