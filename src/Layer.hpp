#pragma once

#include "Matrix.hpp"
#include "LayerType.hpp"

float randomNormalizedFloat() {
	float random = ((float)(rand() % RAND_MAX)) / (float)RAND_MAX;
	float nr = random * 2.0f - 1.0f;
	return nr;
}

class Layer {
public:
	Layer(LayerType layerType, unsigned int outputSize, unsigned int batches) :
		layerType(layerType),
		outputSize(outputSize),
		weights({ layerType.getNeuronCount(), outputSize }),
		biases({ layerType.getNeuronCount() }), 
		weightErrorsSums({ layerType.getNeuronCount(), outputSize, batches }),
		errorsSums({ layerType.getNeuronCount(), batches }),
		outputs({ layerType.getNeuronCount(), batches }),
		inputs({ layerType.getNeuronCount(), batches }),
		errors({ layerType.getNeuronCount(), batches }) {

		setActivationFunction(layerType.getActivationFunction());

		for (int i = 0; i < layerType.getNeuronCount(); i++) {
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
		return layerType.getNeuronCount();
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

	const std::function<float(float)>& getActivationFunction() {
		return activationFunction;
	}

	const std::function<float(float)>& getActivationFunctionDerivative() {
		return activationFunctionDerivative;
	}

	Activation getActivationFunctionType() {
		return layerType.getActivationFunction();
	}

private:
	LayerType layerType;
	unsigned int outputSize;

	Matrix2D<float> weights;
	Matrix1D<float> biases;

	Matrix3D<float> weightErrorsSums;
	Matrix2D<float> errorsSums;

	Matrix2D<float> outputs;
	Matrix2D<float> inputs;

	Matrix2D<float> errors;


	std::function<float(float)> activationFunction;
	std::function<float(float)> activationFunctionDerivative;

	static float sigmoid(float x) {
		return 1.0f / (1.0f + std::exp(-x));
	}

	static float sigmoidDerivative(float x) {
		return sigmoid(x) * (1.0f - sigmoid(x));
	}

	static float relu(float x) {
		return std::max(0.0f, x);
	}

	static float reluDerivative(float x) {
		return (x > 0.0f ? 1.0f : 0.0f);
	}

	static float leakyRelu(float x) {
		return std::max(0.01f * x, x);
	}

	static float leakyReluDerivative(float x) {
		return (x > 0.0f ? 1.0f : 0.01f);
	}

	static float linear(float x) {
		return x;
	}

	static float linearDerivative(float x) {
		return 1.0f;
	}

	void setActivationFunction(Activation activationFunction) {
		switch (activationFunction) {
			case Activation::SIGMOID:
				this->activationFunction = sigmoid;
				this->activationFunctionDerivative = sigmoidDerivative;
				break;
			case Activation::RELU:
				this->activationFunction = relu;
				this->activationFunctionDerivative = reluDerivative;
				break;
			case Activation::LEAKY_RELU:
				this->activationFunction = leakyRelu;
				this->activationFunctionDerivative = leakyReluDerivative;
				break;
			case Activation::LINEAR:
				this->activationFunction = linear;
				this->activationFunctionDerivative = linearDerivative;
				break;
		}
	}
};
