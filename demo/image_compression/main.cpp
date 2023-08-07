#include <iostream>
#include <array>
#include <iomanip>
#include <chrono>

#include "Neuron.hpp"
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

	engine::Display display(width * previewSizeMultiplier, height * previewSizeMultiplier, "Neural Network", false);
	SDL_Event event;

	srand(time(NULL));

	Network network({ 2, 30, 20, 10, 3 });
	network.setLearningRate(0.1f);

	TrainingData trData = { {0.0f, 0.0f}, { 0.0f, 0.0f, 0.0f } };

	const int samplingMultiplier = 1 + imageSize / RAND_MAX;
	for (int iteration = 0; iteration < 1000000000; iteration++) {
		int randomIndex = (rand() * samplingMultiplier) % imageSize;
		trData.outputs[0] = (float)imageData[randomIndex * channels + 0] / 255.0f;
		trData.outputs[1] = (float)imageData[randomIndex * channels + 1] / 255.0f;
		trData.outputs[2] = (float)imageData[randomIndex * channels + 2] / 255.0f;
		trData.inputs[0] = (float)(randomIndex % width) / (float)width;
		trData.inputs[1] = (float)(randomIndex / width) / (float)height;

		network.train(trData, iteration % BATCH_SIZE == BATCH_SIZE - 1);

		if (iteration % 100000 == 0) {
			float error = 0.0f;
			const float previewWidth = width * previewSizeMultiplier;
			const float previewHeight = height * previewSizeMultiplier;
			for (int y = 0; y < previewHeight; y++) {
				for (int x = 0; x < previewWidth; x++) {
					trData.inputs[0] = (float)x / (float)previewWidth;
					trData.inputs[1] = (float)y / (float)previewHeight;
					network.setInputs(trData);
					network.propagateForward();
					error += network.getError(trData);
					float r = network.getOutputLayer()->getNeuron(0)->getOutput();
					float g = network.getOutputLayer()->getNeuron(1)->getOutput();
					float b = network.getOutputLayer()->getNeuron(2)->getOutput();
					hlp::color color = { (unsigned char)(r * 255.0f), (unsigned char)(g * 255.0f), (unsigned char)(b * 255.0f) };
					display.drawPixel(x, y, color);					
				}
			}

			error /= previewWidth * previewHeight;
			std::cout << "Iteration: " << iteration << ", Error: " << std::fixed << error << std::endl;

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
