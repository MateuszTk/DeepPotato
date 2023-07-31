#pragma once

class Neuron {
public:
	Neuron(int outputSize, float bias) {
		if (outputSize > 0) {
			this->outputWeights = new float[outputSize];
			for (int i = 0; i < outputSize; i++) {
				outputWeights[i] = 0.0f;
			}
		}
		else {
			this->outputWeights = nullptr;
		}
		this->outputSize = outputSize;
		this->bias = bias;
		this->input = 0.0f;
		this->output = 0.0f;
		this->error = 0.0f;
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

	void addError(float error) {
		this->error += error;
	}

	float getError() {
		return error;
	}

	~Neuron() {
		delete[] outputWeights;
	}

private:
	float* outputWeights;
	float bias;
	int outputSize;
	float input;
	float output;
	float error;
};
