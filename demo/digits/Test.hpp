#pragma once

#include "Helper.hpp"

class Test {
public:
	Test(const int imWidth, const int imHeight) : 
		width(imWidth),
		height(imHeight),
		imageSize(width* height),
		previewSizeMultiplier((PREVIEW_WIDTH > 0) ? (float)PREVIEW_WIDTH / width : 1.0f),
		display(imWidth, imHeight, "Neural Network", false) {
		
	}

	virtual void run(Network& network) = 0;

protected:
	engine::Display display;

	const int width;
	const int height;
	const int imageSize;
	const float previewSizeMultiplier;
};
