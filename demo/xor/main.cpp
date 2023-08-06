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
	network.setLearningRate(0.1f);

	for (int epoch = 0; epoch < 40000; epoch++) {
		int trDataIndex = epoch % trainingData.size();
		const TrainingData& trData = trainingData[trDataIndex];
		network.train(trData);
		if (epoch % 1000 == 0) {
			float error = network.getError(trData);
			std::cout << "Epoch: " << epoch << ", Error: " << std::fixed << error << std::endl;
		}
	}
	return 0;
}
