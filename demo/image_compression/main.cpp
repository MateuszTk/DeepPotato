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

int main(int argc, char** argv) {
	int width, height, channels;
	unsigned char* imageData = stbi_load("images/input.png", &width, &height, &channels, 0);
	if (imageData == NULL) {
		std::cout << "Failed to load image" << std::endl;
		return 1;
	}

	unsigned int imageSize = width * height;

	engine::Display display(width, height, "Neural Network");

	srand(time(NULL));

	Network network({ 2, 20, 10, 3 });
	network.setLearningRate(0.3f);

	TrainingData trData = { {0.0f, 0.0f}, { 0.0f, 0.0f, 0.0f } };

	for (int epoch = 0; epoch < 1000000000; epoch++) {
		int randomIndex = (rand() * 2) % imageSize;
		trData.outputs[0] = (float)imageData[randomIndex * channels + 0] / 255.0f;
		trData.outputs[1] = (float)imageData[randomIndex * channels + 1] / 255.0f;
		trData.outputs[2] = (float)imageData[randomIndex * channels + 2] / 255.0f;
		trData.inputs[0] = (float)(randomIndex % width) / (float)width;
		trData.inputs[1] = (float)(randomIndex / width) / (float)height;

		network.train(trData);
		if (epoch % 100000 == 0) {
			float error = network.getError(trData);

			std::cout << "Epoch: " << epoch << ", Error: " << std::fixed << error << std::endl;

			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {

					int index = y * width + x;
					trData.inputs[0] = (float)x / (float)width;
					trData.inputs[1] = (float)y / (float)height;
					network.setInputs(trData);
					network.propagateForward();
					float r = network.getOutputLayer()->getNeuron(0)->getOutput();
					float g = network.getOutputLayer()->getNeuron(1)->getOutput();
					float b = network.getOutputLayer()->getNeuron(2)->getOutput();
					hlp::color color = { (unsigned char)(r * 255.0f), (unsigned char)(g * 255.0f), (unsigned char)(b * 255.0f) };
					display.drawPixel(x, y, color);
				}
			}

			SDL_Event event;
			bool quit = false;
			engine::IO::getKeys(&event, &quit);
			display.printFrame();
			if (quit) {
				return 0;
			}
		}
	}
	return 0;
}
