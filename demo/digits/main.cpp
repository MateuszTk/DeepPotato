
// ------------------- CONFIG -------------------
// AUTO_TEST defined: size of the preview window (set to -1 to disable)
// AUTO_TEST undefined: size of the paint canvas
#define PREVIEW_WIDTH 200

// size of the batch
#define BATCH_SIZE 32

// number of threads used for training, set to 0 or not define to disable multithreading
// use multithreading only with big networks, otherwise it will be slower
#define THREAD_POOL_SIZE 4

// enable testing
#define TEST
// enable training
#define TRAIN

// When defined, the program runs a tests and show the results automatically,
//		'*' is the correct answer, '^' is the network's answer
// When undefined, the program allows the user to draw a digits to test the network while it's training
#define AUTO_TEST

// save location (press key 'S')
#define SAVE_PATH "network.dpn"
// load location (press key 'L')
#define LOAD_PATH "network.dpn"

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

#include "IDX_Importer.hpp"

#define SDL_MAIN_HANDLED
#include "engine.h"
#include "Canvas.hpp"

void displayInputImage(engine::Display& display, const TrainingData& image, int width, int height, float previewSizeMultiplier) {
	const float previewWidth = width * previewSizeMultiplier;
	const float previewHeight = height * previewSizeMultiplier;
	for (int y = 0; y < previewHeight; y++) {
		for (int x = 0; x < previewWidth; x++) {
			unsigned char grayScaleColor = image.inputs((int)(y / previewSizeMultiplier) * width + (int)(x / previewSizeMultiplier)) * 255.0f;
			hlp::color color = { grayScaleColor, grayScaleColor, grayScaleColor };
			display.drawPixel(x, y, color);
		}
	}
}

std::pair<hlp::ivec2, hlp::ivec2> boundingBox(const unsigned char* image, int width, int height) {
	hlp::ivec2 minBB = { width, height };
	hlp::ivec2 maxBB = { 0, 0 };
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (image[y * width + x] > 0) {
				if (x < minBB.x) {
					minBB.x = x;
				}
				if (y < minBB.y) {
					minBB.y = y;
				}
				if (x > maxBB.x) {
					maxBB.x = x;
				}
				if (y > maxBB.y) {
					maxBB.y = y;
				}
			}
		}
	}
	return std::make_pair(minBB, maxBB);
}

void autoTest(int iteration, const IDX::IDX_Data& testImages, const IDX::IDX_Data& testLabels, engine::Display& display, Network& network, int width, int height, int imageSize, float previewSizeMultiplier, int testDataIndex);

int main(int argc, char** argv) {
	testMatrix();

	IDX::IDX_Data trainImages = IDX::import("dataset/train-images.idx3-ubyte");
	IDX::printData(trainImages);
	IDX::IDX_Data trainLabels = IDX::import("dataset/train-labels.idx1-ubyte");
	IDX::printData(trainLabels);

#ifdef AUTO_TEST
	IDX::IDX_Data testImages = IDX::import("dataset/t10k-images.idx3-ubyte");
	IDX::printData(testImages);
	IDX::IDX_Data testLabels = IDX::import("dataset/t10k-labels.idx1-ubyte");
	IDX::printData(testLabels);
#endif // AUTO_TEST

	const int width = trainImages.header.sizes[1];
	const int height = trainImages.header.sizes[2];
	const int imageSize = width * height;
	const float previewSizeMultiplier = (PREVIEW_WIDTH > 0) ? (float)PREVIEW_WIDTH / width : 1.0f;

	// calculate bounding boxes for each digit
	std::vector<std::pair<hlp::ivec2, hlp::ivec2>> boundingBoxes(trainImages.header.sizes[0]);
	for (int i = 0; i < trainImages.header.sizes[0]; i++) {
		const unsigned char* image = trainImages.data + i * width * height;
		boundingBoxes[i] = boundingBox(image, width, height);
	}

	engine::Display display(width * previewSizeMultiplier, height * previewSizeMultiplier, "Neural Network", false);

	Canvas canvas(width, height);
	canvas.setScale(previewSizeMultiplier);

	srand(time(NULL));

	Network network({ 28 * 28, 100, 100, 10 });
	network.setLearningRate(0.1f);

	std::vector<TrainingData> trData;
	trData.reserve(BATCH_SIZE);
	for (int i = 0; i < BATCH_SIZE; i++) {
		trData.emplace_back(28 * 28, 10);
	}

	TrainingData testData(28 * 28, 10);

	auto start = std::chrono::high_resolution_clock::now();
	const int samplingMultiplier = 1 + imageSize / RAND_MAX;

	int lastIteration = 0;
	int iteration = -1;
	while (true) {
		iteration++;

#ifdef TRAIN
		int trDataIndex = iteration % trainImages.header.sizes[0];
		const unsigned char* image = trainImages.data + trDataIndex * imageSize;
		const unsigned char label = trainLabels.data[trDataIndex];

		// apply random offset to digit image, so that the network can learn to recognize digits which are not centered
		std::pair<hlp::ivec2, hlp::ivec2> bb = boundingBoxes[trDataIndex];
		int randomOffsetX = ((rand() % width) - width / 2 + 1) * 0.5f;
		int randomOffsetY = ((rand() % height) - height / 2 + 1) * 0.5f;

		randomOffsetX = ((randomOffsetX + bb.second.x < width) ? randomOffsetX : (width - bb.second.x - 1));
		randomOffsetY = ((randomOffsetY + bb.second.y < height) ? randomOffsetY : (height - bb.second.y - 1));

		randomOffsetX = ((randomOffsetX + bb.first.x >= 0) ? randomOffsetX : (-bb.first.x));
		randomOffsetY = ((randomOffsetY + bb.first.y >= 0) ? randomOffsetY : (-bb.first.y));

		int batchSample = iteration % BATCH_SIZE;

		// set inputs
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int randomX = x - randomOffsetX;
				int randomY = y - randomOffsetY;
				if (randomX < 0 || randomX >= width || randomY < 0 || randomY >= height) {
					trData[batchSample].inputs(y * width + x) = 0.0f;
					continue;
				}
				trData[batchSample].inputs(y * width + x) = (float)image[randomY * width + randomX] / 255.0f;
			}
		}
		// set outputs
		for (int i = 0; i < 10; i++) {
			trData[batchSample].outputs(i) = (i == label) ? 1.0f : 0.0f;
		}

		if (iteration % BATCH_SIZE == BATCH_SIZE - 1) {
			// train network
			network.trainBatch(trData);
		}

#ifdef TEST
#ifdef AUTO_TEST
		const int testDelay = 4000;
#else // !AUTO_TEST
		const int testDelay = 100;
#endif // AUTO_TEST
#else // !TEST
		const int testDelay = 5000;
#endif // TEST

#else // !TRAIN
		const int testDelay = 1;
#endif // TRAIN


		if (iteration % testDelay == 0) {
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> diff = end - start;

			if (diff.count() >= 50) {
				start = end;
#ifdef TEST
#ifdef AUTO_TEST
				int testDataIndex = (iteration / testDelay) % testImages.header.sizes[0];
				autoTest(iteration, testImages, testLabels, display, network, width, height, imageSize, previewSizeMultiplier, testDataIndex);

				std::cout << "Training speed: " << (iteration - lastIteration) / diff.count() * 1000 << " iterations per second\n";
				lastIteration = iteration;
#ifndef TRAIN
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
#endif // !TRAIN
#else // !AUTO_TEST

				//displayInputImage(display, trData, width, height, previewSizeMultiplier);

				// allow user to draw a digit on the canvas
				canvas.paint();
				canvas.drawCanvas(display);

				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {
						testData.inputs(y * width + x) = canvas.getPixel({x, y}).r / 255.0f;
					}
				}

				// feed input to network
				network.setInputs(testData, 0);
				network.propagateForward(0);

				// find output with highest value
				int maxIndex = 0;
				float maxOutput = network.getOutputLayer()->getOutputs()(0, 0);
				for (int i = 1; i < 10; i++) {
					float output = network.getOutputLayer()->getOutputs()(i, 0);
					if (output > maxOutput) {
						maxOutput = output;
						maxIndex = i;
					}
				}

				// print output
#ifdef TRAIN
				std::cout << "Iteration: " << iteration << '\n';
#endif // TRAIN
				for (int i = 0; i < 10; i++) {
					std::cout << i << " " << std::fixed << std::setprecision(2) << network.getOutputLayer()->getOutputs()(i, 0) << " ";
					if (i == maxIndex) {
						std::cout << " ^";
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
#endif // AUTO_TEST
#else // !TEST
				std::cout << "Iteration: " << iteration << " ,";
				std::cout << "Training speed: " << (iteration - lastIteration) / diff.count() * 1000 << " iterations per second\n";
				lastIteration = iteration;
#endif // TEST

				// handle user keyboard input
				bool quit = false;
				SDL_Event event;
				const Uint8* keystates = engine::IO::getKeys(&event, &quit);
				if (keystates[SDL_SCANCODE_ESCAPE]) {
					quit = true;
				}
				if (keystates[SDL_SCANCODE_SPACE]) {
					canvas.clear();
				}
				if (keystates[SDL_SCANCODE_S]) {
					network.save(SAVE_PATH);
				}
				if (keystates[SDL_SCANCODE_L]) {
					network.load(LOAD_PATH);
				}

				// display canvas
				display.printFrame();

				if (quit) {
					return 0;
				}
			}
		}
	}
	return 0;
}

void autoTest(int iteration, const IDX::IDX_Data& testImages, const IDX::IDX_Data& testLabels, engine::Display& display, Network& network, int width, int height, int imageSize, float previewSizeMultiplier, int testDataIndex) {
	static TrainingData trData(28 * 28, 10);

	// Test network
	const unsigned char* timage = testImages.data + testDataIndex * imageSize;
	const unsigned char tlabel = testLabels.data[testDataIndex];

	for (int i = 0; i < 28 * 28; i++) {
		trData.inputs(i) = (float)timage[i] / 255.0f;
	}

	for (int i = 0; i < 10; i++) {
		trData.outputs(i) = (i == tlabel) ? 1.0f : 0.0f;
	}

	const unsigned int testSlot = 0;
	network.setInputs(trData, testSlot);
	network.propagateForward(testSlot);

	displayInputImage(display, trData, width, height, previewSizeMultiplier);

	float error = network.getError(trData, testSlot);

	int maxIndex = 0;
	float maxOutput = network.getOutputLayer()->getOutputs()(0, testSlot);
	for (int i = 1; i < 10; i++) {
		float output = network.getOutputLayer()->getOutputs()(i, testSlot);
		if (output > maxOutput) {
			maxOutput = output;
			maxIndex = i;
		}
	}

	std::cout << "Iteration: " << iteration << ", TestId: " << testDataIndex << ", Error: " << std::fixed << std::setprecision(8) << error << std::endl;
	for (int i = 0; i < 10; i++) {
		std::cout << i << " " << std::fixed << std::setprecision(2) << network.getOutputLayer()->getOutputs()(i, testSlot) << " ";
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
}
