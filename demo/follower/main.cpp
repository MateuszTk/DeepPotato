
// ------------------- CONFIG -------------------
	// --------TRAINING--------
		// number of threads used for training, set to 0 or not define to disable multithreading
			#define THREAD_POOL_SIZE 4
		// number of training data per batch
			#define BATCH_SIZE 32

	// --------DEMO--------
		#define USE_NETWORK

// ----------------------------------------------

#include <iostream>
#include <array>
#include <iomanip>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <thread>

#include "ThreadPool.hpp"
#include "Layer.hpp"
#include "Network.hpp"

#define SDL_MAIN_HANDLED
#include "engine.h"
#include "Canvas.hpp"

#include "LineFollower.hpp"

int main(int argc, char** argv) {
	SDL_Event event;
	bool quit = false;
	double deltaTime = 0.0;
	double currTime = 0.0;
	double lastTime = 0.0;

	LineFollower follower;

	srand(time(NULL));
	Network network({
		InputLayer(7),
		DenseLayer(6, Activation::RELU),
		DenseLayer(6, Activation::RELU),
		DenseLayer(3, Activation::SIGMOID)
	});
	network.setLearningRate(0.05f);

	std::vector<TrainingData> trainingData;
	trainingData.reserve(BATCH_SIZE);
	for (int i = 0; i < BATCH_SIZE; i++) {
		trainingData.emplace_back(7, 3);
	}
	int batchIndex = 0;
	bool pause = true;

	while (true) {
		const Uint8* keys = engine::IO::getKeys(&event, &quit);
		if (quit) break;
		if (keys[SDL_SCANCODE_SPACE]) pause = !pause;
		if (keys[SDL_SCANCODE_R]) { follower.reset(); batchIndex = 0; }
		if (keys[SDL_SCANCODE_C]) follower.clear();
		if (keys[SDL_SCANCODE_D]) follower.action(0, 1.0 * deltaTime);
		if (keys[SDL_SCANCODE_A]) follower.action(1, 1.0 * deltaTime);
		if (keys[SDL_SCANCODE_W]) follower.action(2, 40.0 * deltaTime);
		if (keys[SDL_SCANCODE_S]) follower.action(3, 40.0 * deltaTime);

		follower.updateCanvas();

		float reward = follower.getReward();
		Matrix1D<float> state = follower.sampleLine(true);

		/*std::cout << std::fixed << std::setprecision(2);
		for (int i = 0; i < state.getDimension(0); i++) {
			std::cout << state(i) << ' ';
		}
		std::cout << std::endl;*/

		//std::cout << "reward: " << reward << std::endl;
		float simulationSpeed = 2.0f;

		if (!pause) {
			if ((currTime - lastTime) * simulationSpeed >= 0.1) {
#ifdef USE_NETWORK
				// use deep q learning
				TrainingData data(7, 3);
				data.inputs = state;
				network.setInputs(data, 0);
				network.propagateForward(0);
				Matrix1D<float> output = *network.getOutputLayer()->getOutputs()(0);
				int action = 0;
				for (int i = 1; i < output.getDimension(0); i++) {
					if (output(i) > output(action)) action = i;
				}
				data.outputs = output;

				if (action != 2) {
					if (action == 0) follower.action(0, 0.1);
					if (action == 1) follower.action(1, 0.1);
				}

				if (batchIndex >= BATCH_SIZE) {
					//std::cout << "Training...\n";
					network.trainBatch(trainingData);
					batchIndex = 0;

					static const char actions[] = { 'R', 'L', 'F'};
					for (int i = 0; i < follower.getSensorCount(); i++) {
						for (int j = 0; j < follower.getSensorCount(); j++) {
							data.inputs(j) = (i == j ? 1.0f : 0.0f);
						}
						network.setInputs(data, 0);
						network.propagateForward(0);
						auto outputs = network.getOutputLayer()->getOutputs()(0);
						int actionMax = 0;
						for (int j = 1; j < outputs->getDimension(0); j++) {
							if (outputs->operator()(j) > outputs->operator()(actionMax)) actionMax = j;
						}
						std::cout << actions[actionMax] << ' ';
					}
					std::cout << std::endl;

					// check if the car is out of the track
					if (reward <= 0.0f) {
						follower.reset();
						batchIndex = 0;
					}
				}
				else {
					trainingData[batchIndex] = data;
					if (batchIndex > 0) {
						//std::cout << "reward: " << reward << std::endl;
						trainingData[batchIndex - 1].outputs(action) = (0.1f * reward + 0.9f * trainingData[batchIndex].outputs(action));
						//trainingData[batchIndex - 1].outputs(action) = (reward + 0.96f * trainingData[batchIndex].outputs(action));
					}
					batchIndex++;
				}
#else
				if (state(1) > 0.0) follower.action(1, 0.1);
				if (state(state.getDimension(0) - 2) > 0.0) follower.action(0, 0.1);
#endif
				lastTime = currTime;
				}
			follower.action(2, 40.0 * deltaTime * simulationSpeed);
		}

		deltaTime = follower.print();
		currTime += deltaTime;
	}
	
	return 0;
}
