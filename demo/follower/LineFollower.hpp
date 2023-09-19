#pragma once

#include "Canvas.hpp"

class LineFollower {
public:
	LineFollower() : display(800, 800, "Line Follower", false), canvas(display.getScreenWidth(), display.getScreenHeight()), sensorCount(7) {
		canvas.setBrushSize(4);
		canvas.clear();

		actor.surface = engine::IO::openTexture("textures/car.bmp");
		actor.position = { 0.0, 0.0 };
		actor.rotation = 0.0;
		scene.addActor(&actor);
	}

	Matrix1D<float> sampleLine(bool draw) {
		Matrix1D<float> line({ (unsigned int)sensorCount });
		for (int i = 0; i < sensorCount; i++) {
			line(i) = samplePoint({ 30.0f - i * 10.0f, -50.0f }, draw, 2);
		}
		return line;
	}

	float getReward() {
		//highest reward is when the car is in the middle of the line
		Matrix1D<float> line = sampleLine(false);
		int indexSum = 0;
		int indexCount = 0;
		for (int i = 0; i < sensorCount; i++) {
			if (line(i) > 0.0f) {
				indexSum += i;
				indexCount++;
			}
		}
		float reward = 0.0f;
		float index = indexSum / (float)indexCount;
		if (indexCount != 0) reward = (3 - std::abs(index - 3) + 1) / 4.0;
		return std::powf(std::clamp(reward, 0.0f, 1.0f), 2.0f);
	}

	void updateCanvas() {
		canvas.paint();
		canvas.drawCanvas(display);
	}

	double print() {
		scene.drawActors(&display);
		return display.printFrame();
	}

	void reset() {
		actor.position = { 0.0, 0.0 };
		actor.rotation = 0.0;
	}

	void clear() {
		canvas.clear();
	}

	int getSensorCount() {
		return sensorCount;
	}

	void action(int action, double speed) {
		switch (action) {
		case 0:
			actor.rotation += speed;
			break;
		case 1:
			actor.rotation -= speed;
			break;
		case 2:
			actor.move({ 0.0, speed });
			break;
		case 3:
			actor.move({ 0.0, -speed });
			break;
		}
	}

private:
	engine::Display display;
	Canvas canvas;
	engine::Scene scene;
	engine::Actor actor;
	const int sensorCount;

	float samplePoint(hlp::fvec2 offset, bool draw, int size) {
		hlp::fvec2 origin = {
				(float)(actor.position.x + display.getScreenWidth() / 2),
				(float)(-actor.position.y + display.getScreenHeight() / 2)
		};
		hlp::fvec2 samplePos = origin + offset;
		samplePos = hlp::fvec2::rotatePoint(samplePos, origin, -actor.rotation);

		float value = 0.0f;
		for (int y = -size; y < size + 1; y++) {
			for (int x = -size; x < size + 1; x++) {
				if (samplePos.x + x >= 0 && samplePos.x + x < display.getScreenWidth() &&
					samplePos.y + y >= 0 && samplePos.y + y < display.getScreenHeight()) {
					value += display.getPixel(samplePos.x + x, samplePos.y + y).r;
					if (draw) {
						display.drawPixel(samplePos.x + x, samplePos.y + y, hlp::color(255, 0, 0));
					}
				}
			}
		}

		return value / (hlp::pow2(1 + 2 * size) * 255.0f);
	}
};
