#pragma once

#include "Test.hpp"

class CanvasTest : public Test {
public:
	CanvasTest() : 
		Test(28, 28),
		canvas(width, height) {

		canvas.setScale(previewSizeMultiplier);
	}

	void run(Network& network) override {
		// allow user to draw a digit on the canvas
		canvas.paint();
		canvas.drawCanvas(display);

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				testData.inputs(y * width + x) = canvas.getPixel({ x, y }).r / 255.0f;
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
		for (int i = 0; i < 10; i++) {
			std::cout << i << " " << std::fixed << std::setprecision(2) << network.getOutputLayer()->getOutputs()(i, 0) << " ";
			if (i == maxIndex) {
				std::cout << " ^";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;

		Test::run(network);
	}

	void clearCanvas() {
		canvas.clear();
	}

private:
	Canvas canvas;
};
