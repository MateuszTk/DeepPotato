#include <iostream>
#include <array>
#include <iomanip>
#include <chrono>

#define THREAD_POOL_SIZE 4

#include "Layer.hpp"
#include "Network.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define SDL_MAIN_HANDLED
#include "engine.h"

#define PREVIEW_WIDTH -1

#define BATCH_SIZE 32

int main(int argc, char** argv) {
	int width, height, channels;
	unsigned char* imageData = stbi_load("images/input.png", &width, &height, &channels, 0);
	if (imageData == NULL) {
		std::cout << "Failed to load image" << std::endl;
		return 1;
	}

	const unsigned int imageSize = width * height;
	const float previewSizeMultiplier = (PREVIEW_WIDTH > 0) ? (float)PREVIEW_WIDTH / width : 1.0f;

	engine::Display display(width * previewSizeMultiplier, height * previewSizeMultiplier, "Learned image", false);
	SDL_Event event;

	srand(time(NULL));

	Network network({ 2, 30, 20, 10, 3 });
	network.setLearningRate(0.1f);

	std::vector<TrainingData> trData;
	trData.reserve(BATCH_SIZE);
	for (int i = 0; i < BATCH_SIZE; i++) {
		trData.emplace_back(2, 3);
	}

	auto start = std::chrono::high_resolution_clock::now();

	const int samplingMultiplier = 1 + imageSize / RAND_MAX;
	for (int iteration = 0; iteration < 1000000000; iteration++) {
		int trDataIndex = iteration % BATCH_SIZE;

		int randomIndex = (rand() * samplingMultiplier) % imageSize;
		trData[trDataIndex].outputs(0) = (float)imageData[randomIndex * channels + 0] / 255.0f;
		trData[trDataIndex].outputs(1) = (float)imageData[randomIndex * channels + 1] / 255.0f;
		trData[trDataIndex].outputs(2) = (float)imageData[randomIndex * channels + 2] / 255.0f;
		trData[trDataIndex].inputs(0) = (float)(randomIndex % width) / (float)width;
		trData[trDataIndex].inputs(1) = (float)(randomIndex / width) / (float)height;

		if (trDataIndex == BATCH_SIZE - 1) {
			network.trainBatch(trData);
		}

		if (iteration % 100000 == 0) {
			float error = 0.0f;
			const float previewWidth = width * previewSizeMultiplier;
			const float previewHeight = height * previewSizeMultiplier;
			for (int y = 0; y < previewHeight; y++) {
				for (int x = 0; x < previewWidth; x++) {
					trData[0].inputs(0) = (float)x / (float)previewWidth;
					trData[0].inputs(1) = (float)y / (float)previewHeight;
					network.setInputs(trData[0], 0);
					network.propagateForward(0);
					error += network.getError(trData[0], 0);
					float r = network.getOutputLayer()->getOutputs()(0, 0);
					float g = network.getOutputLayer()->getOutputs()(1, 0);
					float b = network.getOutputLayer()->getOutputs()(2, 0);
					hlp::color color = { (unsigned char)(r * 255.0f), (unsigned char)(g * 255.0f), (unsigned char)(b * 255.0f) };
					display.drawPixel(x, y, color);					
				}
			}

			error /= previewWidth * previewHeight;
			std::cout << "Iteration: " << iteration << ", Error: " << std::fixed << error;

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> elapsed = end - start;
			std::cout << " | " << (100000) / elapsed.count() * 1000 << " iterations per second\n";
			start = end;

			bool quit = false;
			const Uint8* keystates = engine::IO::getKeys(&event, &quit);
			if (keystates[SDL_SCANCODE_ESCAPE]) {
				quit = true;
			}

			display.printFrame();
			
			if (quit) {
				return 0;
			}
		}
	}
	return 0;
}
