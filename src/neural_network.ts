interface DenseMLP {
    layers: number[];
    weights: Float32Array;
    biases: Float32Array;
}

class MLP implements DenseMLP {
    layers: number[];
    weights: Float32Array;
    biases: Float32Array;

    constructor(layers: number[]) {
        this.layers = layers;

        let weightCount = 0;
        let biasCount = 0;

        for (let i = 1; i < layers.length; i++) {
            weightCount += layers[i - 1] * layers[i];
            biasCount += layers[i];
        }

        this.weights = new Float32Array(weightCount);
        this.biases = new Float32Array(biasCount);
    }
}