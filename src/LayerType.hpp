#pragma once

enum class Activation {
	SIGMOID,
	RELU,
	LEAKY_RELU,
	LINEAR
};

class LayerType {
public:	
	LayerType(unsigned int neuronCount, Activation activation) : neuronCount(neuronCount), activation(activation) {

	}

	unsigned int getNeuronCount() const {
		return neuronCount;
	}

	Activation getActivationFunction() const {
		return activation;
	}

protected:
	unsigned int neuronCount;
	Activation activation;
};

class InputLayer : public LayerType {
public:
	InputLayer(unsigned int inputSize) : LayerType(inputSize, Activation::LINEAR) {

	}
};

class DenseLayer : public LayerType {
public:
	DenseLayer(unsigned int neuronCount, Activation activation) : LayerType(neuronCount, activation) {

	}	
};
