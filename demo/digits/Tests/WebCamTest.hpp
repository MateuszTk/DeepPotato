#pragma once

#include "Test.hpp"
#include "Webcam.hpp"

class WebCamTest : public Test {
public:
	WebCamTest(const int camId) :
		Test(28, 28),
		webcam(camId) {

	}

	void run(Network& network) override {
		Mat frame = webcam.getFrame();
		if (frame.empty()) {
			std::cout << "Failed to get frame from webcam\n";
			return;
		}

		std::vector<std::pair<Rect, Mat>> digits = webcam.findDigits(frame);
		std::vector<int> labels;

		for (std::pair<Rect, Mat>& digit : digits) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					testData.inputs(y * width + x) = digit.second.at<unsigned char>(y, x) / 255.0f;
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

			labels.push_back(maxIndex);
			std::cout << maxIndex << " ";
		}
		displayInputImage(display, testData, width, height, previewSizeMultiplier);
		std::cout << '\n';

		Test::run(network);
	}

private:
	Webcam webcam;
};
