export type ActivationFunction = (value: number) => number;

export interface DenseMLPInterface {
    layers: number[];
    inputNumber: number;
    weights: Float32Array;
    biases: Float32Array;
    activationFunction: ActivationFunction;
    output: Float32Array;
}

export class DenseMLP implements DenseMLPInterface {
    layers: number[];
    inputNumber: number;
    weights: Float32Array;
    biases: Float32Array;
    activationFunction: ActivationFunction;
    output: Float32Array;

    constructor(
        layers: number[],
        weights: Float32Array,
        biases: Float32Array,
        activationFunction: ActivationFunction = Math.tanh,
    ) {
        const inputNumber: number | undefined = layers[0]
        if (inputNumber === undefined) {
            throw new Error("There are no layers passed!")
        }
        const outputSize: number | undefined = layers.at(-1);
        const numberOfLayers: number = layers.length
        if (outputSize === undefined || numberOfLayers < 2) {
            throw new Error("MLP requires at least 2 layers");
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

        this.layers = [...layers];
        this.weights = weights;
        this.biases = biases;
        this.activationFunction = activationFunction;
        this.output = new Float32Array(outputSize);
        this.inputNumber = inputNumber;
    }
    forwardPass(input: Float32Array): Float32Array {
        if (input.length !== this.inputNumber) {
            throw new Error(`Expected ${this.inputNumber} inputs; received ${input.length}.`);
        }

        let currentInput: Float32Array = input;
        let nextInput: Float32Array = new Float32Array;
        let weightTraversalOffset: number = 0;
        let biasTraversalOffset: number = 0;
        const numberOfLayers: number = this.layers.length;
        const maxLayerIndex: number = numberOfLayers - 1;

        for (let currentLayerIndex = 0; currentLayerIndex < maxLayerIndex; currentLayerIndex++) {
            const currentLayerSize: number | undefined = this.layers[currentLayerIndex];
            if (currentLayerSize == undefined) {
                throw new Error("There is no current layer to iterate over!");
            }
            const nextLayerSize: number | undefined = this.layers[currentLayerIndex + 1];
            if (nextLayerSize == undefined) {
                throw new Error("There is no next layer to iterate over!");
            }

            nextInput = new Float32Array(nextLayerSize);
            for (let nextLayerNeuronIndex = 0; nextLayerNeuronIndex < nextLayerSize; nextLayerNeuronIndex++) {
                let weightedSum: number = 0;
                for (let currentLayerNeuronIndex = 0; currentLayerNeuronIndex < currentLayerSize; currentLayerNeuronIndex++) {
                    const activation: number | undefined = currentInput[currentLayerNeuronIndex];
                    const weight: number | undefined = this.weights[weightTraversalOffset];
                    if (activation === undefined || weight === undefined) {
                        throw new Error("Activation or weight is undefined")
                    }

                    weightedSum += activation * weight;
                    weightTraversalOffset++;
                }
                const bias: number | undefined = this.biases[biasTraversalOffset];
                if (bias === undefined) {
                    throw new Error("Undefined bias!");
                }

                const activationFunctionInput: number = weightedSum + bias;
                nextInput[nextLayerNeuronIndex] = this.activationFunction(activationFunctionInput);
                biasTraversalOffset++;
            }
            currentInput = nextInput;
        }
        return nextInput;
    }
}