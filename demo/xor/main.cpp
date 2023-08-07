#include <iostream>
#include <array>
#include <iomanip>
#include <chrono>

#include "Neuron.hpp"
#include "Layer.hpp"
#include "Network.hpp"

std::array<TrainingData, 4> trainingData = { {
	{{0.0f, 0.0f}, { 0.0f }},
	{{0.0f, 1.0f}, { 1.0f }},
	{{1.0f, 0.0f}, { 1.0f }},
	{{1.0f, 1.0f}, { 0.0f }}
} };

int main() {
	srand(time(NULL));

	Network network({ 2, 3, 1 });
	network.setLearningRate(1.0f);

	for (int iteration = 0; iteration < 4000; iteration++) {
		int trDataIndex = iteration % trainingData.size();
		const TrainingData& trData = trainingData[trDataIndex];
		bool endOfBatch = (iteration % trainingData.size()) == (trainingData.size() - 1);
		network.train(trData, endOfBatch);
		if (iteration % 100 == 0) {
			float error = network.getError(trData);
			std::cout << "Iteration: " << iteration << ", Error: " << std::fixed << error << std::endl;
		}
	}
	return 0;
}
