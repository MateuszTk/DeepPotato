#pragma once

class Neuron {
public:
	Neuron(int outputSize, float bias) {
		if (outputSize > 0) {
			this->outputWeights = new float[outputSize];
			for (int i = 0; i < outputSize; i++) {
				outputWeights[i] = 0.0f;
			}
			this->weightErrorsSums = new float[outputSize];
			for (int i = 0; i < outputSize; i++) {
				weightErrorsSums[i] = 0.0f;
			}
		}
		else {
			this->outputWeights = nullptr;
			this->weightErrorsSums = nullptr;
		}
		this->outputSize = outputSize;
		this->bias = bias;
		this->input = 0.0f;
		this->output = 0.0f;
		this->error = 0.0f;
		this->errorSum = 0.0f;
	}

	float* getOutputWeights() {
		return outputWeights;
	}

	float getBias() {
		return bias;
	}

	void setBias(float bias) {
		this->bias = bias;
	}

	int getOutputSize() {
		return outputSize;
	}

	void setInput(float input) {
		this->input = input;
	}

	float getInput() {
		return input;
	}

	void setOutput(float output) {
		this->output = output;
	}

	float getOutput() {
		return output;
	}

	void setError(float error) {
		this->error = error;
	}

	float getError() {
		return error;
	}

	void setErrorSum(float errorSum) {
		this->errorSum = errorSum;
	}

	void addErrorSum(float errorSum) {
		this->errorSum += errorSum;
	}

	float getErrorSum() {
		return errorSum;
	}

	void setWeightErrorSum(int iWeight, float weightErrorSum) {
		this->weightErrorsSums[iWeight] = weightErrorSum;
	}

	void addWeightErrorSum(int iWeight, float weightError) {
		this->weightErrorsSums[iWeight] += weightError;
	}

	float getWeightErrorSum(int iWeight) {
		return weightErrorsSums[iWeight];
	}

	void resetWeightErrorSums() {
		for (int i = 0; i < outputSize; i++) {
			weightErrorsSums[i] = 0.0f;
		}
	}

	~Neuron() {
		delete[] outputWeights;
		delete[] weightErrorsSums;
	}

private:
	float* outputWeights;
	float* weightErrorsSums;
	float bias;
	int outputSize;
	float input;
	float output;
	float error;
	float errorSum;
};
