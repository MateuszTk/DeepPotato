#include <iostream>
#include <array>
#include <iomanip>
#include <chrono>

#include "Layer.hpp"
#include "Network.hpp"

std::array<TrainingData, 4> trainingData = {{
	{{0.0f, 0.0f}, { 0.0f }},
	{{0.0f, 1.0f}, { 1.0f }},
	{{1.0f, 0.0f}, { 1.0f }},
	{{1.0f, 1.0f}, { 0.0f }}
} };

int main() {
	srand(time(NULL));

	Network network({ 
		InputLayer(2),
		DenseLayer(3, Activation::SIGMOID),
		DenseLayer(1, Activation::SIGMOID)
	});
	network.setLearningRate(1.0f);

	float error = 0.0f;

	for (int iteration = 0; iteration < 4000; iteration++) {
		int trDataIndex = iteration % trainingData.size();
		const TrainingData& trData = trainingData[trDataIndex];
		bool endOfBatch = (iteration % trainingData.size()) == (trainingData.size() - 1);
		network.train(trData, endOfBatch, 0);
		if (iteration % 100 < 4) {
			error += network.getError(trData, 0);
			if (iteration % 100 == 3) {
				std::cout << "Iteration: " << iteration + 1 << ", Error: " << std::fixed << error / 4.0f << std::endl;
				error = 0.0f;
			}
		}
	}

	std::cout << std::endl;

	for (const TrainingData& trData : trainingData) {
		network.setInputs(trData, 0);
		network.propagateForward(0);
		auto output = network.getOutputLayer()->getOutputs();
		std::cout << "Input: " << trData.inputs(0) << ' ' << trData.inputs(1) << ", Output: " << output(0, 0) << std::endl;
	}

	return 0;
}
