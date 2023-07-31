#pragma once

#include "Neuron.hpp"

float randomNormalizedFloat() {
	float random = ((float)(rand() % RAND_MAX)) / (float)RAND_MAX;
	float nr = random * 2.0f - 1.0f;
	return nr;
}

class Layer {
public:
	Layer(int neuronCount, int outputSize) {
		this->neuronCount = neuronCount;
		this->neurons = new Neuron * [neuronCount];
		for (int i = 0; i < neuronCount; i++) {
			Neuron* neuron = new Neuron(outputSize, 0.0f);
			neurons[i] = neuron;
			neuron->setBias(randomNormalizedFloat());
			for (int j = 0; j < outputSize; j++) {
				neuron->getOutputWeights()[j] = randomNormalizedFloat();
			}
		}
	}

	Neuron** getNeurons() {
		return neurons;
	}

	Neuron* getNeuron(int n) {
		return neurons[n];
	}

	int getNeuronCount() {
		return neuronCount;
	}

	~Layer() {
		for (int i = 0; i < neuronCount; i++) {
			delete neurons[i];
		}
		delete[] neurons;
	}
private:
	Neuron** neurons;
	int neuronCount;
};
