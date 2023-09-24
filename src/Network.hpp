#pragma once

#include <initializer_list>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cassert>

#include "Matrix.hpp"
#include "ThreadPool.hpp"
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
	Network(const std::initializer_list<LayerType>& layersTypes) :
		learningRate(0.1f), 
		threadPool(THREAD_POOL_SIZE),
		batchSize(std::max<int>(THREAD_POOL_SIZE, 1)) {

		this->layerCount = layersTypes.size();
		this->layers = new Layer*[layerCount];
		int i = 0;
		for (auto layerT = layersTypes.begin(); layerT < layersTypes.end(); layerT++) {

			int nextLayerSize = (i + 1 >= layerCount) ? 0 : (layerT + 1)->getNeuronCount();
			Layer* layer = new Layer(*layerT, nextLayerSize, batchSize);
			layers[i] = layer;
			i++;
		}
		resetErrorSums();
	}

	void setInputs(const TrainingData& data, unsigned int batch) {
		*layers[0]->getInputs()(batch) = data.inputs;
		*layers[0]->getOutputs()(batch) = data.inputs;
	}

	void propagateForward(unsigned int batch) {
		for (int layer = 1; layer < layerCount; layer++) {
			Layer* currentLayer = layers[layer];
			Layer* previousLayer = layers[layer - 1];

			multiplyAndAdd(previousLayer->getWeights(), *previousLayer->getOutputs()(batch), currentLayer->getBiases(), *currentLayer->getInputs()(batch));
			currentLayer->getOutputs().applyFunction(currentLayer->getInputs(), currentLayer->getActivationFunction());
		}
	}

	void propagateError(const TrainingData& targetData, unsigned int batch) {
		for (int layer = layerCount - 1; layer > 0; layer--) {
			Layer* currentLayer = layers[layer];
			Layer* nextLayer = layers[layer + 1];
			Layer* previousLayer = layers[layer - 1];
			
			for (int iNeuron = 0; iNeuron < currentLayer->getNeuronCount(); iNeuron++) {
				float errorSum = 0.0f;
				if (layer == layerCount - 1) {
					errorSum = targetData.outputs(iNeuron) - currentLayer->getOutputs()(iNeuron, batch);
				}
				else{
					for (int iNextNeuron = 0; iNextNeuron < nextLayer->getNeuronCount(); iNextNeuron++) {
						errorSum += nextLayer->getErrors()(iNextNeuron, batch) * currentLayer->getWeights()(iNeuron, iNextNeuron);
					}
				}
				errorSum *= currentLayer->getActivationFunctionDerivative()(currentLayer->getInputs()(iNeuron, batch));
				currentLayer->getErrors()(iNeuron, batch) = errorSum;

				// sum errors for bias and weights
				currentLayer->getErrorsSums()(iNeuron, batch) += errorSum;
				for (int iPrevNeuron = 0; iPrevNeuron < previousLayer->getNeuronCount(); iPrevNeuron++) {
					previousLayer->getWeightErrorsSums()(iPrevNeuron, iNeuron, batch) += errorSum * previousLayer->getOutputs()(iPrevNeuron, batch);
				}
			}
		}
	}

	void updateWeightsAndBiases() {
		for (int batch = 0; batch < batchSize; batch++) {
			for (int layer = 1; layer < layerCount; layer++) {
				Layer* currentLayer = layers[layer];
				Layer* previousLayer = layers[layer - 1];

				for (int iNeuron = 0; iNeuron < currentLayer->getNeuronCount(); iNeuron++) {
					for (int iPrevNeuron = 0; iPrevNeuron < previousLayer->getNeuronCount(); iPrevNeuron++) {
						float delta = previousLayer->getWeightErrorsSums()(iPrevNeuron, iNeuron, batch);
						previousLayer->getWeights()(iPrevNeuron, iNeuron) += delta * this->learningRate;
					}
					currentLayer->getBiases()(iNeuron) += currentLayer->getErrorsSums()(iNeuron, batch) * this->learningRate;
				}
			}
		}
	}

	void resetErrorSums() {
		for (int batch = 0; batch < batchSize; batch++) {
			for (int layer = 0; layer < layerCount; layer++) {
				Layer* currentLayer = layers[layer];
				for (int iNeuron = 0; iNeuron < currentLayer->getNeuronCount(); iNeuron++) {
					currentLayer->getErrorsSums()(iNeuron, batch) = 0.0f;
					for (int iWeight = 0; iWeight < currentLayer->getOutputSize(); iWeight++) {
						currentLayer->getWeightErrorsSums()(iNeuron, iWeight, batch) = 0.0f;
					}
				}
			}
		}
	}

	void train(const TrainingData& data, bool endOfBatch, unsigned int batchId) {
		setInputs(data, batchId);
		propagateForward(batchId);
		propagateError(data, batchId);
		if (endOfBatch) {
			updateWeightsAndBiases();
			resetErrorSums();
		}
	}

	void trainBatch(const std::vector<TrainingData>& data) {
		if (THREAD_POOL_SIZE > 0) {
			for (int i = 0; i < data.size(); i++) {
				threadPool.addJob([this, &data, i](int a, int threadId) {
					train(data[i], false, threadId);
				}, 1);
			}
			threadPool.wait();
		}
		else {
			for (int i = 0; i < data.size(); i++) {
				train(data[i], false, 0);
			}
		}

		updateWeightsAndBiases();
		resetErrorSums();
	}

	float getError(const TrainingData& data, unsigned int batch) {
		float error = 0.0f;
		const int neuronCount = layers[layerCount - 1]->getNeuronCount();
		for (int i = 0; i < neuronCount; i++) {
			float delta = data.outputs(i) - layers[layerCount - 1]->getOutputs()(i, batch);
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
				if (weightCount > 0) {
					for (int iWeight = 0; iWeight < weightCount; iWeight++) {
						float weight = layer->getWeights()(iNeuron, iWeight);
						file.write((char*)&weight, sizeof(float));
					}
				}
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

			LayerType type(neuronCount, Activation::SIGMOID);
			layers[iLayer] = new Layer(type, weightCount, batchSize);

			std::cout << "Layer " << iLayer << ": " << neuronCount << " neurons, " << weightCount << " weights\n";

			for (int iNeuron = 0; iNeuron < neuronCount; iNeuron++) {
				float bias;
				file.read((char*)&bias, sizeof(float));
				layers[iLayer]->getBiases()(iNeuron) = bias;
				if (weightCount > 0) {
					for (int iWeight = 0; iWeight < weightCount; iWeight++) {
						float* weight = &(layers[iLayer]->getWeights()(iNeuron, iWeight));
						file.read((char*)weight, sizeof(float));
					}
				}
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
	unsigned int batchSize;
};
