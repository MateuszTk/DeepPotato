#pragma once

#include "Test.hpp"

class AutoTest : public Test {
public:
	AutoTest(const char* testImagesSrc, const char* testLabelsSrc) :
		testImages(IDX::import(testImagesSrc)),
		testLabels(IDX::import(testLabelsSrc)),
		Test(28, 28),
		testDataIndex(0) {

		IDX::printData(this->testImages);
		IDX::printData(this->testLabels);
	}

	void run(Network& network) override {
		// Test network
		const unsigned char* timage = testImages.data + testDataIndex * imageSize;
		const unsigned char tlabel = testLabels.data[testDataIndex];

		for (int i = 0; i < 28 * 28; i++) {
			testData.inputs(i) = (float)timage[i] / 255.0f;
		}

		for (int i = 0; i < 10; i++) {
			testData.outputs(i) = (i == tlabel) ? 1.0f : 0.0f;
		}

		const unsigned int testSlot = 0;
		network.setInputs(testData, testSlot);
		network.propagateForward(testSlot);

		displayInputImage(display, testData, width, height, previewSizeMultiplier);

		float error = network.getError(testData, testSlot);

		int maxIndex = 0;
		float maxOutput = network.getOutputLayer()->getOutputs()(0, testSlot);
		for (int i = 1; i < 10; i++) {
			float output = network.getOutputLayer()->getOutputs()(i, testSlot);
			if (output > maxOutput) {
				maxOutput = output;
				maxIndex = i;
			}
		}

		//std::cout << "Iteration: " << iteration << ", TestId: " << testDataIndex << ", Error: " << std::fixed << std::setprecision(8) << error << std::endl;
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

		testDataIndex = (testDataIndex + 1) % testImages.header.sizes[0];

		Test::run(network);
	}

private:
	const IDX::IDX_Data testImages;
	const IDX::IDX_Data testLabels;

	int testDataIndex;
};
