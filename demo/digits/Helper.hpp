#pragma once

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
