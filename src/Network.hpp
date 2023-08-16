#pragma once

#include <initializer_list>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

#include "ThreadPool.hpp"
#include "Neuron.hpp"
#include "Layer.hpp"

#ifndef THREAD_POOL_SIZE
#define THREAD_POOL_SIZE 0
#endif

struct TrainingData {
	std::vector<float> inputs;
	std::vector<float> outputs;
};

class Network {
public:
	Network(const std::initializer_list<int>& layersSizes) : learningRate(0.1f), threadPool(THREAD_POOL_SIZE) {
		this->layerCount = layersSizes.size();
		this->layers = new Layer*[layerCount];
		int i = 0;
		for (auto it = layersSizes.begin(); it < layersSizes.end(); it++) {
			int nextLayerSize = (i + 1 >= layerCount) ? 0 : *(it + 1);
			Layer* layer = new Layer(*it, nextLayerSize);
			layers[i] = layer;
			i++;
		}
		resetErrorSums();
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
			
			auto job = std::function<void(unsigned int)>([this, currentLayer, previousLayer](unsigned int iNeuron) {
				Neuron* neuron = currentLayer->getNeuron(iNeuron);
				float sum = 0.0f;
				for (int j = 0; j < previousLayer->getNeuronCount(); j++) {
					Neuron* previousNeuron = previousLayer->getNeuron(j);
					sum += previousNeuron->getOutput() * previousNeuron->getOutputWeights()[iNeuron];
				}
				neuron->setInput(sum);
				neuron->setOutput(sigmoid(sum + neuron->getBias()));
			});

			if (THREAD_POOL_SIZE <= 0) {
				for (int iNeuron = 0; iNeuron < currentLayer->getNeuronCount(); iNeuron++) {
					job(iNeuron);
				}
			}
			else {
				threadPool.addJob(job, currentLayer->getNeuronCount());			
				threadPool.wait();
			}
		}
	}

	void propagateError(const TrainingData& targetData) {
		for (int layer = layerCount - 1; layer > 0; layer--) {
			Layer* currentLayer = layers[layer];
			Layer* nextLayer = layers[layer + 1];
			Layer* previousLayer = layers[layer - 1];
			
			for (int iNeuron = 0; iNeuron < currentLayer->getNeuronCount(); iNeuron++) {
				Neuron* neuron = currentLayer->getNeuron(iNeuron);
				float errorSum = 0.0f;
				if (layer == layerCount - 1) {
					errorSum = targetData.outputs[iNeuron] - neuron->getOutput();
				}
				else{
					for (int iNextNeuron = 0; iNextNeuron < nextLayer->getNeuronCount(); iNextNeuron++) {
						errorSum += nextLayer->getNeuron(iNextNeuron)->getError() * neuron->getOutputWeights()[iNextNeuron];
					}
				}
				errorSum *= sigmoidDerivative(neuron->getInput() + neuron->getBias());
				neuron->setError(errorSum);

				// sum errors for bias and weights
				neuron->addErrorSum(errorSum);
				for (int iPrevNeuron = 0; iPrevNeuron < previousLayer->getNeuronCount(); iPrevNeuron++) {
					Neuron* previousNeuron = previousLayer->getNeuron(iPrevNeuron);
					previousNeuron->addWeightErrorSum(iNeuron, errorSum * previousNeuron->getOutput());
				}
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
					float delta = previousNeuron->getWeightErrorSum(iNeuron);
					previousNeuron->getOutputWeights()[iNeuron] += delta * this->learningRate;
				}
				neuron->setBias(neuron->getBias() + neuron->getErrorSum() * this->learningRate);
			}
		}
	}

	void resetErrorSums() {
		for (int layer = 0; layer < layerCount; layer++) {
			Layer* currentLayer = layers[layer];
			for (int iNeuron = 0; iNeuron < currentLayer->getNeuronCount(); iNeuron++) {
				Neuron* neuron = currentLayer->getNeuron(iNeuron);
				neuron->setErrorSum(0.0f);
				neuron->resetWeightErrorSums();
			}
		}
	}

	void train(const TrainingData& data, bool endOfBatch) {
		setInputs(data);
		propagateForward();
		propagateError(data);
		if (endOfBatch) {
			updateWeightsAndBiases();
			resetErrorSums();
		}
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

	void save(const char* path) {
		std::ofstream file(path, std::ios::binary);
		if (!file.is_open()) {
			std::cout << "Failed to open file " << path << '\n';
			return;
		}

		file.write((char*)&layerCount, sizeof(int));
		for (int iLayer = 0; iLayer < layerCount; iLayer++) {
			Layer* layer = layers[iLayer];

			int neuronCount = layer->getNeuronCount();
			file.write((char*)&neuronCount, sizeof(int));

			int weightCount = 0;
			if (layer->getNeuronCount() > 0) {
				weightCount = layer->getNeuron(0)->getOutputSize();
			}
			file.write((char*)&weightCount, sizeof(int));

			for (int iNeuron = 0; iNeuron < neuronCount; iNeuron++) {
				Neuron* neuron = layer->getNeuron(iNeuron);

				float bias = neuron->getBias();
				file.write((char*)&bias, sizeof(float));
				
				file.write((char*)(neuron->getOutputWeights()), sizeof(float) * weightCount);
			}
		}

		file.close();
		std::cout << "Saved network to " << path << '\n';
	}

	void load(const char* path) {
		for (int i = 0; i < layerCount; i++) {
			delete layers[i];
		}
		delete[] layers;

		std::ifstream file(path, std::ios::binary);

		file.read((char*)&layerCount, sizeof(int));
		layers = new Layer*[layerCount];
		for (int iLayer = 0; iLayer < layerCount; iLayer++) {
			int neuronCount;
			file.read((char*)&neuronCount, sizeof(int));

			int weightCount;
			file.read((char*)&weightCount, sizeof(int));

			layers[iLayer] = new Layer(neuronCount, weightCount);

			for (int iNeuron = 0; iNeuron < neuronCount; iNeuron++) {
				Neuron* neuron = layers[iLayer]->getNeuron(iNeuron);

				float bias;
				file.read((char*)&bias, sizeof(float));
				neuron->setBias(bias);

				file.read((char*)(neuron->getOutputWeights()), sizeof(float) * weightCount);
			}
		}

		file.close();
		std::cout << "Loaded network from " << path << '\n';
	}

	~Network() {
		for (int i = 0; i < layerCount; i++) {
			delete layers[i];
		}
		delete[] layers;
	}

private:
	ThreadPool threadPool;

	Layer** layers;
	int layerCount;
	float learningRate;

	float sigmoid(float x) {
		return 1.0f / (1.0f + std::exp(-x));
	}

	float sigmoidDerivative(float x) {
		return sigmoid(x) * (1.0f - sigmoid(x));
	}
};
