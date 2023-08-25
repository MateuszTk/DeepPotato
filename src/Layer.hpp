#pragma once

#include "Matrix.hpp"

float randomNormalizedFloat() {
	float random = ((float)(rand() % RAND_MAX)) / (float)RAND_MAX;
	float nr = random * 2.0f - 1.0f;
	return nr;
}

class Layer {
public:
	Layer(unsigned int neuronCount, unsigned int outputSize, unsigned int batches)
		: neuronCount(neuronCount), outputSize(outputSize), weights({ neuronCount, outputSize }), biases({ neuronCount }), weightErrorsSums({ neuronCount, outputSize, batches }),
		errorsSums({ neuronCount, batches }), outputs({ neuronCount, batches }), inputs({ neuronCount, batches }), errors({ neuronCount, batches }) {


		for (int i = 0; i < neuronCount; i++) {
			biases(i) = randomNormalizedFloat();
			for (int j = 0; j < outputSize; j++) {
				weights(i, j) = randomNormalizedFloat();
				for (int k = 0; k < batches; k++) {
					weightErrorsSums(i, j, k) = 0.0f;
				}
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

	Matrix3D<float>& getWeightErrorsSums() {
		return weightErrorsSums;
	}

	Matrix2D<float>& getErrorsSums() {
		return errorsSums;
	}

	Matrix2D<float>& getOutputs() {
		return outputs;
	}

	Matrix2D<float>& getInputs() {
		return inputs;
	}

	Matrix2D<float>& getErrors() {
		return errors;
	}

private:
	unsigned int neuronCount;
	unsigned int outputSize;

	Matrix2D<float> weights;
	Matrix1D<float> biases;

	Matrix3D<float> weightErrorsSums;
	Matrix2D<float> errorsSums;

	Matrix2D<float> outputs;
	Matrix2D<float> inputs;

	Matrix2D<float> errors;
};
