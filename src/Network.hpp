#pragma once

#include <initializer_list>
#include <vector>

#include "Neuron.hpp"
#include "Layer.hpp"

struct TrainingData {
	std::vector<float> inputs;
	std::vector<float> outputs;
};

class Network {
public:
	Network(const std::initializer_list<int>& layersSizes) : learningRate(0.1f) {
		this->layerCount = layersSizes.size();
		this->layers = new Layer*[layerCount];
		int i = 0;
		for (auto it = layersSizes.begin(); it < layersSizes.end(); it++) {
			int nextLayerSize = (i + 1 >= layerCount) ? 0 : *(it + 1);
			Layer* layer = new Layer(*it, nextLayerSize);
			layers[i] = layer;
			i++;
		}
	}

	void setInputs(const TrainingData& data) {
		for (int i = 0; i < layers[0]->getNeuronCount(); i++) {
			Neuron* neuron = layers[0]->getNeuron(i);
			neuron->setInput(data.inputs[i]);
			neuron->setOutput(data.inputs[i]);
		}
	}

	void propagateForward() {
		for (int layer = 1; layer < layerCount; layer++) {
			Layer* currentLayer = layers[layer];
			Layer* previousLayer = layers[layer - 1];

			for (int iNeuron = 0; iNeuron < currentLayer->getNeuronCount(); iNeuron++) {
				Neuron* neuron = currentLayer->getNeuron(iNeuron);
				float sum = 0.0f;
				for (int j = 0; j < previousLayer->getNeuronCount(); j++) {
					Neuron* previousNeuron = previousLayer->getNeuron(j);
					sum += previousNeuron->getOutput() * previousNeuron->getOutputWeights()[iNeuron];
				}
				neuron->setInput(sum);
				neuron->setOutput(sigmoid(sum + neuron->getBias()));
			}
		}
	}

	void propagateError(const TrainingData& targetData) {
		for (int i = 0; i < layers[layerCount - 1]->getNeuronCount(); i++) {
			Neuron* neuron = layers[layerCount - 1]->getNeuron(i);
			float delta = targetData.outputs[i] - neuron->getOutput();
			neuron->setError(delta * sigmoidDerivative(neuron->getInput() + neuron->getBias()));
		}

		for (int layer = layerCount - 2; layer > 0; layer--) {
			Layer* currentLayer = layers[layer];
			Layer* nextLayer = layers[layer + 1];

			for (int iNeuron = 0; iNeuron < currentLayer->getNeuronCount(); iNeuron++) {
				Neuron* neuron = currentLayer->getNeuron(iNeuron);
				float errorSum = 0.0f;
				for (int iNextNeuron = 0; iNextNeuron < nextLayer->getNeuronCount(); iNextNeuron++) {
					errorSum += nextLayer->getNeuron(iNextNeuron)->getError() * neuron->getOutputWeights()[iNextNeuron];
				}
				errorSum *= sigmoidDerivative(neuron->getInput() + neuron->getBias());
				neuron->setError(errorSum);
			}
		}
	}

	void updateWeightsAndBiases() {
		for (int layer = 1; layer < layerCount; layer++) {
			Layer* currentLayer = layers[layer];
			Layer* previousLayer = layers[layer - 1];

			for (int iNeuron = 0; iNeuron < currentLayer->getNeuronCount(); iNeuron++) {
				Neuron* neuron = currentLayer->getNeuron(iNeuron);
				for (int iPrevNeuron = 0; iPrevNeuron < previousLayer->getNeuronCount(); iPrevNeuron++) {
					Neuron* previousNeuron = previousLayer->getNeuron(iPrevNeuron);
					float delta = previousNeuron->getOutput() * neuron->getError();
					previousNeuron->getOutputWeights()[iNeuron] += delta * this->learningRate;
				}
				neuron->setBias(neuron->getBias() + neuron->getError() * this->learningRate);
			}
		}
	}

	void train(const TrainingData& data) {
		setInputs(data);
		propagateForward();
		propagateError(data);
		updateWeightsAndBiases();
	}

	float getError(const TrainingData& data) {
		float error = 0.0f;
		const int neuronCount = layers[layerCount - 1]->getNeuronCount();
		for (int i = 0; i < neuronCount; i++) {
			Neuron* neuron = layers[layerCount - 1]->getNeuron(i);
			float delta = data.outputs[i] - neuron->getOutput();
			error += delta * delta;
		}
		return error / neuronCount;
	}

	Layer* getOutputLayer() {
		return layers[layerCount - 1];
	}

	void setLearningRate(float learningRate) {
		this->learningRate = learningRate;
	}

	float getLearningRate() {
		return learningRate;
	}

	~Network() {
		for (int i = 0; i < layerCount; i++) {
			delete layers[i];
		}
		delete[] layers;
	}

private:
	Layer** layers;
	int layerCount;
	float learningRate;

	float sigmoid(float x) {
		return 1.0f / (1.0f + exp(-x));
	}

	float sigmoidDerivative(float x) {
		return sigmoid(x) * (1.0f - sigmoid(x));
	}
};
