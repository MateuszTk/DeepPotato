#pragma once
#include "engine.h"

class Canvas {
public:
	Canvas(unsigned int width, unsigned int height) : width(width), height(height) {
		this->pixels = new unsigned char[width * height * 3];
		clear();
	}

	void paint() {
		hlp::ivec2 mousePos;
		if (SDL_GetMouseState(&mousePos.x, &mousePos.y) & SDL_BUTTON_LMASK) {
			mousePos -= this->position;
			hlp::fvec2 fMousePos = mousePos;
			fMousePos.x /= scale;
			fMousePos.y /= scale;
			if (fMousePos.x >= 0 && fMousePos.x < this->width && fMousePos.y >= 0 && fMousePos.y < this->height) {
				if (antialiasing) {
					for (int i = -1; i < 2; i++) {
						for (int j = -1; j < 2; j++) {
							hlp::ivec2 pos = fMousePos + hlp::ivec2(i, j);
							if (pos.x >= 0 && pos.x < this->width && pos.y >= 0 && pos.y < this->height) {
								hlp::fvec2 delta = hlp::fvec2(pos) - fMousePos;
								float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);
								float intensity = std::clamp(1.2f - dist, 0.0f, 1.0f);
								hlp::color pixColor = {
									(unsigned char)(brushColor.r * intensity),
									(unsigned char)(brushColor.g * intensity),
									(unsigned char)(brushColor.b * intensity)
								};
							
								hlp::color currPix = getPixel(pos);
								pixColor.r = std::max(pixColor.r, currPix.r);
								pixColor.g = std::max(pixColor.g, currPix.g);
								pixColor.b = std::max(pixColor.b, currPix.b);						
								this->setPixel(pos, pixColor);
							}
						}
					}
				}
				hlp::ivec2 pos = fMousePos;
				if (pos.x >= 0 && pos.x < this->width && pos.y >= 0 && pos.y < this->height) {
					this->setPixel(pos, this->brushColor);
				}
			}
		}
	}

	void setBrushColor(const hlp::color& color) {
		this->brushColor = color;
	}

	const hlp::color& getBrushColor() const {
		return this->brushColor;
	}

	void drawCanvas(engine::Display& display) {
		for (int y = 0; y < this->height; y++) {
			for (int x = 0; x < this->width; x++) {
				hlp::color pixColor = hlp::color(
					this->pixels[(y * this->width + x) * 3 + 0],
					this->pixels[(y * this->width + x) * 3 + 1],
					this->pixels[(y * this->width + x) * 3 + 2]
				);
				// draw square as pixel
				for (int i = 0; i < scale; i++) {
					for (int j = 0; j < scale; j++) {
						display.drawPixel(this->position.x + x * scale + i, this->position.y + y * scale + j, pixColor);
					}
				}
			}
		}
	}

	void clear() {
		memset(this->pixels, 0, this->width * this->height * 3);
	}

	void setPosition(const hlp::ivec2& position) {
		this->position = position;
	}

	const hlp::ivec2& getPosition() const {
		return this->position;
	}

	void setScale(float scale) {
		this->scale = scale;
	}

	float getScale() const {
		return this->scale;
	}

	void setPixel(const hlp::ivec2& pos, const hlp::color& color) {
		this->pixels[(pos.y * this->width + pos.x) * 3 + 0] = color.r;
		this->pixels[(pos.y * this->width + pos.x) * 3 + 1] = color.g;
		this->pixels[(pos.y * this->width + pos.x) * 3 + 2] = color.b;
	}

	hlp::color getPixel(const hlp::ivec2& pos) const {
		return hlp::color(
			this->pixels[(pos.y * this->width + pos.x) * 3 + 0],
			this->pixels[(pos.y * this->width + pos.x) * 3 + 1],
			this->pixels[(pos.y * this->width + pos.x) * 3 + 2]
		);
	}

	void setAntialiasing(bool antialiasing) {
		this->antialiasing = antialiasing;
	}

	bool getAntialiasing() const {
		return this->antialiasing;
	}

	~Canvas() {
		delete[] this->pixels;
	}

private:
	unsigned int width;
	unsigned int height;
	unsigned char* pixels;
	hlp::color brushColor = {255, 255, 255};
	hlp::ivec2 position = {0, 0};
	float scale = 1.0f;
	bool antialiasing = true;
};
