
// ------------------- CONFIG -------------------
	// --------TESTING--------
		// enable testing
			#define TEST

		// When defined, the program runs a tests and show the results automatically,
		//		'*' is the correct answer, '^' is the network's answer
		// When undefined, the program allows the user to draw a digits to test the network while it's training
			#define AUTO_TEST

		// When defined, the program uses the webcam as input
			//#define WEBCAM

	// --------TRAINING--------
		// enable training
			#define TRAIN

		// size of the batch
			#define BATCH_SIZE 32

		// number of threads used for training, set to 0 or not define to disable multithreading
			#define THREAD_POOL_SIZE 4

	// --------OTHER--------
		// AUTO_TEST defined: size of the preview window (set to -1 to disable)
		// AUTO_TEST undefined: size of the paint canvas
			#define PREVIEW_WIDTH 200

		// save location (press key 'S')
			#define SAVE_PATH "network.dpn"
		// load location (press key 'L')
			#define LOAD_PATH "network.dpn"

		// delay between each test iteration (in ms)
		#ifdef AUTO_TEST
			#define TEST_DELAY 1000
		#else
			#ifdef WEBCAM
				#define TEST_DELAY 50
			#else
				#define TEST_DELAY 50
			#endif
		#endif

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

#include "Helper.hpp"

#include "Test.hpp"
#include "AutoTest.hpp"
#ifdef WEBCAM
#include "Webcam.hpp"
#include "WebCamTest.hpp"
#endif
#include "CanvasTest.hpp"

int main(int argc, char** argv) {

	IDX::IDX_Data trainImages = IDX::import("dataset/train-images.idx3-ubyte");
	IDX::printData(trainImages);
	IDX::IDX_Data trainLabels = IDX::import("dataset/train-labels.idx1-ubyte");
	IDX::printData(trainLabels);

	const int width = trainImages.header.sizes[1];
	const int height = trainImages.header.sizes[2];
	const int imageSize = width * height;

	// calculate bounding boxes for each digit
	std::vector<std::pair<hlp::ivec2, hlp::ivec2>> boundingBoxes(trainImages.header.sizes[0]);
	for (int i = 0; i < trainImages.header.sizes[0]; i++) {
		const unsigned char* image = trainImages.data + i * width * height;
		boundingBoxes[i] = boundingBox(image, width, height);
	}

#ifdef TEST
	#ifdef AUTO_TEST
		Test* test = new AutoTest("dataset/t10k-images.idx3-ubyte", "dataset/t10k-labels.idx1-ubyte");
	#else
		#ifdef WEBCAM
			Test* test = new WebCamTest(0);
		#else
			Test* test = new CanvasTest();
		#endif
	#endif
#endif

	srand(time(NULL));

	Network network({ 28 * 28, 100, 100, 10 });
	network.setLearningRate(0.1f);

	std::vector<TrainingData> trData;
	trData.reserve(BATCH_SIZE);
	for (int i = 0; i < BATCH_SIZE; i++) {
		trData.emplace_back(28 * 28, 10);
	}

	auto start = std::chrono::high_resolution_clock::now();
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
		hlp::ivec2 randomOffsetV = randomOffset(width, height, bb);

		int batchSample = iteration % BATCH_SIZE;

		// set inputs
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int randomX = x - randomOffsetV.x;
				int randomY = y - randomOffsetV.y;
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
#endif // TRAIN

		if (iteration % 100 == 0) {
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> diff = end - start;

			if (diff.count() >= TEST_DELAY) {
				start = end;
#ifdef TEST
				test->run(network);

#ifdef TRAIN
				std::cout << "Training speed: " << (iteration - lastIteration) / diff.count() * 1000 << " iterations per second\n";
				lastIteration = iteration;
#endif // TRAIN

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
					// clear canvas if space is pressed and is instance of CanvasTest
					CanvasTest* canvasTest = dynamic_cast<CanvasTest*>(test);
					if (canvasTest != nullptr) {
						canvasTest->clearCanvas();
					}
				}
				if (keystates[SDL_SCANCODE_S]) {
					network.save(SAVE_PATH);
				}
				if (keystates[SDL_SCANCODE_L]) {
					network.load(LOAD_PATH);
				}
				if (quit) {
					return 0;
				}
			}
		}
	}

#ifdef TEST
	delete test;
#endif // TEST	

	return 0;
}
