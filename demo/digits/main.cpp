#include <iostream>
#include <array>
#include <iomanip>
#include <chrono>

#include "Neuron.hpp"
#include "Layer.hpp"
#include "Network.hpp"

#include "IDX_Importer.hpp"

#define SDL_MAIN_HANDLED
#include "engine.h"

#define PREVIEW_WIDTH 100

#define BATCH_SIZE 32

int main(int argc, char** argv) {
	IDX::IDX_Data trainImages = IDX::import("dataset/train-images.idx3-ubyte");
	IDX::printData(trainImages);
	IDX::IDX_Data trainLabels = IDX::import("dataset/train-labels.idx1-ubyte");
	IDX::printData(trainLabels);

	IDX::IDX_Data testImages = IDX::import("dataset/t10k-images.idx3-ubyte");
	IDX::printData(testImages);
	IDX::IDX_Data testLabels = IDX::import("dataset/t10k-labels.idx1-ubyte");
	IDX::printData(testLabels);

	const unsigned int width = trainImages.header.sizes[1];
	const unsigned int height = trainImages.header.sizes[2];
	const unsigned int imageSize = width * height;
	const float previewSizeMultiplier = (PREVIEW_WIDTH > 0) ? (float)PREVIEW_WIDTH / width : 1.0f;

	engine::Display display(width * previewSizeMultiplier, height * previewSizeMultiplier, "Neural Network", false);
	SDL_Event event;

	srand(time(NULL));

	Network network({ 28 * 28, 100, 10 });
	network.setLearningRate(0.1f);

	TrainingData trData;
	trData.inputs.resize(28 * 28);
	trData.outputs.resize(10);

	const int samplingMultiplier = 1 + imageSize / RAND_MAX;
	for (int iteration = 0; iteration < 1000000000; iteration++) {
		int trDataIndex = iteration % trainImages.header.sizes[0];
		const unsigned char* image = trainImages.data + trDataIndex * imageSize;
		const unsigned char label = trainLabels.data[trDataIndex];

		for (int i = 0; i < 28 * 28; i++) {
			trData.inputs[i] = (float)image[i] / 255.0f;
		}

		for (int i = 0; i < 10; i++) {
			trData.outputs[i] = (i == label) ? 1.0f : 0.0f;
		}

		network.train(trData, iteration % BATCH_SIZE == BATCH_SIZE - 1);

		if (iteration % 5000 == 0) {
			// Test network
			int testDataIndex = (iteration / 1000) % testImages.header.sizes[0];
			const unsigned char* timage = testImages.data + testDataIndex * imageSize;
			const unsigned char tlabel = testLabels.data[testDataIndex];

			for (int i = 0; i < 28 * 28; i++) {
				trData.inputs[i] = (float)timage[i] / 255.0f;
			}

			for (int i = 0; i < 10; i++) {
				trData.outputs[i] = (i == tlabel) ? 1.0f : 0.0f;
			}

			network.setInputs(trData);
			network.propagateForward();

			const float previewWidth = width * previewSizeMultiplier;
			const float previewHeight = height * previewSizeMultiplier;
			for (int y = 0; y < previewHeight; y++) {
				for (int x = 0; x < previewWidth; x++) {
					unsigned char grayScaleColor = trData.inputs[(int)(y / previewSizeMultiplier) * width + (int)(x / previewSizeMultiplier)] * 255.0f;
					hlp::color color = { grayScaleColor, grayScaleColor, grayScaleColor };
					display.drawPixel(x, y, color);					
				}
			}

			float error = network.getError(trData);

			int maxIndex = 0;
			float maxOutput = network.getOutputLayer()->getNeuron(0)->getOutput();
			for (int i = 1; i < 10; i++) {
				float output = network.getOutputLayer()->getNeuron(i)->getOutput();
				if (output > maxOutput) {
					maxOutput = output;
					maxIndex = i;
				}
			}

			std::cout << "Iteration: " << iteration << ", TestId: " << testDataIndex << ", Error: " << std::fixed << std::setprecision(8) << error << std::endl;
			for (int i = 0; i < 10; i++) {
				std::cout << i << " " << std::fixed << std::setprecision(2) << network.getOutputLayer()->getNeuron(i)->getOutput() << " ";
				if (i == tlabel) {
					std::cout << " *";
				}
				else {
					std::cout << "  ";
				}
				if (i == maxIndex) {
					std::cout << " ^";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;

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
