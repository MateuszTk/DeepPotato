#pragma once

//#include "Neuron.hpp"
#include "Matrix.hpp"

float randomNormalizedFloat() {
	float random = ((float)(rand() % RAND_MAX)) / (float)RAND_MAX;
	float nr = random * 2.0f - 1.0f;
	return nr;
}

class Layer {
public:
	Layer(unsigned int neuronCount, unsigned int outputSize)
		: neuronCount(neuronCount), outputSize(outputSize), weights({ outputSize, neuronCount }), biases({ neuronCount }), weightErrorsSums({ outputSize, neuronCount }),
		errorsSums({ neuronCount }), outputs({ neuronCount }), inputs({ neuronCount }), errors({ neuronCount }) {


		for (int i = 0; i < neuronCount; i++) {
			biases(i) = randomNormalizedFloat();
			for (int j = 0; j < outputSize; j++) {
				weights(j, i) = randomNormalizedFloat();
			}
		}
	}

	unsigned int getNeuronCount() {
		return neuronCount;
	}

	unsigned int getOutputSize() {
		return outputSize;
	}

	Matrix2D<float>& getWeights() {
		return weights;
	}

	Matrix1D<float>& getBiases() {
		return biases;
	}

	Matrix2D<float>& getWeightErrorsSums() {
		return weightErrorsSums;
	}

	Matrix1D<float>& getErrorsSums() {
		return errorsSums;
	}

	Matrix1D<float>& getOutputs() {
		return outputs;
	}

	Matrix1D<float>& getInputs() {
		return inputs;
	}

	Matrix1D<float>& getErrors() {
		return errors;
	}

private:
	unsigned int neuronCount;
	unsigned int outputSize;

	Matrix2D<float> weights;
	Matrix1D<float> biases;

	Matrix2D<float> weightErrorsSums;
	Matrix1D<float> errorsSums;

	Matrix1D<float> outputs;
	Matrix1D<float> inputs;

	Matrix1D<float> errors;
};
