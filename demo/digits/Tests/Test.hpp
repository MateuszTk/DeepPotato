#pragma once

#include "Helper.hpp"

class Test {
public:
	Test(const int imWidth, const int imHeight) : 
		width(imWidth),
		height(imHeight),
		imageSize(imWidth * imHeight),
		previewSizeMultiplier((PREVIEW_WIDTH > 0) ? (float)PREVIEW_WIDTH / imWidth : 1.0f),
		display(imWidth * previewSizeMultiplier, imHeight * previewSizeMultiplier, "Neural Network", false),
		testData(imWidth * imHeight, 10) {
		
	}

	virtual void run(Network& network) {
		display.printFrame();
	}

protected:
	const int width;
	const int height;
	const int imageSize;
	const float previewSizeMultiplier;

	engine::Display display;
	TrainingData testData;
};
