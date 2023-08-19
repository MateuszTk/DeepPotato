#pragma once

#include <initializer_list>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

#include "ThreadPool.hpp"
//#include "Neuron.hpp"
#include "Layer.hpp"

#ifndef THREAD_POOL_SIZE
#define THREAD_POOL_SIZE 0
#endif

struct TrainingData {
	Matrix1D<float> inputs;
	Matrix1D<float> outputs;

	TrainingData(unsigned int inputSize, unsigned int outputSize) : inputs({ inputSize }), outputs({ outputSize }) {}
	TrainingData(const std::initializer_list<float>& inputs, const std::initializer_list<float>& outputs) : inputs({ (unsigned int)inputs.size() }), outputs({ (unsigned int)outputs.size() }) {
		int i = 0;
		for (auto it = inputs.begin(); it < inputs.end(); it++) {
			this->inputs(i) = *it;
			i++;
		}
		i = 0;
		for (auto it = outputs.begin(); it < outputs.end(); it++) {
			this->outputs(i) = *it;
			i++;
		}
	}
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
		layers[0]->getInputs() = data.inputs;
		layers[0]->getOutputs() = data.inputs;
	}

	void propagateForward() {
		for (int layer = 1; layer < layerCount; layer++) {
			Layer* currentLayer = layers[layer];
			Layer* previousLayer = layers[layer - 1];
			
			auto job = std::function<void(unsigned int)>([this, currentLayer, previousLayer](unsigned int iNeuron) {
				float sum = currentLayer->getBiases()(iNeuron);
				for (int j = 0; j < previousLayer->getNeuronCount(); j++) {
					sum += previousLayer->getOutputs()(j) * previousLayer->getWeights()(iNeuron, j);
				}
				currentLayer->getInputs()(iNeuron) = sum;
				currentLayer->getOutputs()(iNeuron) = sigmoid(sum);
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
				float errorSum = 0.0f;
				if (layer == layerCount - 1) {
					errorSum = targetData.outputs(iNeuron) - currentLayer->getOutputs()(iNeuron);
				}
				else{
					for (int iNextNeuron = 0; iNextNeuron < nextLayer->getNeuronCount(); iNextNeuron++) {
						errorSum += nextLayer->getErrors()(iNextNeuron) * currentLayer->getWeights()(iNextNeuron, iNeuron);
					}
				}
				errorSum *= sigmoidDerivative(currentLayer->getInputs()(iNeuron));
				currentLayer->getErrors()(iNeuron) = errorSum;

				// sum errors for bias and weights
				currentLayer->getErrorsSums()(iNeuron) += errorSum;
				for (int iPrevNeuron = 0; iPrevNeuron < previousLayer->getNeuronCount(); iPrevNeuron++) {
					previousLayer->getWeightErrorsSums()(iNeuron, iPrevNeuron) += errorSum * previousLayer->getOutputs()(iPrevNeuron);
				}
			}
		}
	}

	void updateWeightsAndBiases() {
		for (int layer = 1; layer < layerCount; layer++) {
			Layer* currentLayer = layers[layer];
			Layer* previousLayer = layers[layer - 1];

			for (int iNeuron = 0; iNeuron < currentLayer->getNeuronCount(); iNeuron++) {
				for (int iPrevNeuron = 0; iPrevNeuron < previousLayer->getNeuronCount(); iPrevNeuron++) {
					float delta = previousLayer->getWeightErrorsSums()(iNeuron, iPrevNeuron);
					previousLayer->getWeights()(iNeuron, iPrevNeuron) += delta * this->learningRate;
				}
				currentLayer->getBiases()(iNeuron) += currentLayer->getErrorsSums()(iNeuron) * this->learningRate;
			}
		}
	}

	void resetErrorSums() {
		for (int layer = 0; layer < layerCount; layer++) {
			Layer* currentLayer = layers[layer];
			for (int iNeuron = 0; iNeuron < currentLayer->getNeuronCount(); iNeuron++) {
				currentLayer->getErrorsSums()(iNeuron) = 0.0f;
				for (int iWeight = 0; iWeight < currentLayer->getOutputSize(); iWeight++) {
					currentLayer->getWeightErrorsSums()(iWeight, iNeuron) = 0.0f;
				}
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
			float delta = data.outputs(i) - layers[layerCount - 1]->getOutputs()(i);
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
				weightCount = layer->getOutputSize();
			}
			file.write((char*)&weightCount, sizeof(int));

			for (int iNeuron = 0; iNeuron < neuronCount; iNeuron++) {
				float bias = layer->getBiases()(iNeuron);
				file.write((char*)&bias, sizeof(float));
				
				file.write((char*)(layer->getWeights().dataAt(0, iNeuron)), sizeof(float) * weightCount);
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
				float bias;
				file.read((char*)&bias, sizeof(float));
				layers[iLayer]->getBiases()(iNeuron) = bias;

				file.read((char*)(layers[iLayer]->getWeights().dataAt(0, iNeuron)), sizeof(float) * weightCount);
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
