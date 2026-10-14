export type ActivationFunction = (value: number) => number;

export interface DenseMLP {
    layers: number[];
    weights: Float32Array;
    biases: Float32Array;
    activationFunction: ActivationFunction;
    output: Float32Array;
}

export class MLP implements DenseMLP {
    layers: number[];
    weights: Float32Array;
    biases: Float32Array;
    activationFunction: ActivationFunction;
    output: Float32Array;

    constructor(
        layers: number[],
        activationFunction: ActivationFunction = Math.tanh,
    ) {
        if (layers.length < 2) {
            throw new Error(`MLP requires at least 2 layers; received ${layers.length}.`);
        }

        if (layers.some(size => !Number.isInteger(size) || size <= 0)) {
            throw new Error("Every layer must have a positive integer size.");
        }

        let weightCount = 0;
        let biasCount = 0;

        for (let index = 1; index < layers.length; index++) {
            const inputSize = layers[index - 1];
            const outputSize = layers[index];

            if (inputSize === undefined || outputSize === undefined) {
                throw new Error("Every layer must have a defined size.");
            }

            weightCount += inputSize * outputSize;
            biasCount += outputSize;
        }

        const outputSize = layers.at(-1);
        if (outputSize === undefined) {
            throw new Error("MLP requires a defined output layer.");
        }

        this.layers = [...layers];
        this.weights = new Float32Array(weightCount);
        this.biases = new Float32Array(biasCount);
        this.activationFunction = activationFunction;
        this.output = new Float32Array(outputSize);
    }
}