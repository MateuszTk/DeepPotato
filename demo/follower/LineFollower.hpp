#pragma once

#include "Canvas.hpp"

class LineFollower {
public:
	LineFollower() : display(800, 800, "Line Follower", false), canvas(display.getScreenWidth(), display.getScreenHeight()) {
		canvas.setBrushSize(4);
		canvas.clear();

		actor.surface = engine::IO::openTexture("textures/car.bmp");
		actor.position = { 0.0, 0.0 };
		actor.rotation = 0.0;
		scene.addActor(&actor);
	}

	Matrix1D<float> sampleLine() {
		Matrix1D<float> line({ 6 });
		for (int i = 0; i < 6; i++) {
			line(i) = samplePoint({ 25.0f - i * 10.0f, -50.0f });
		}
		return line;
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
		canvas.clear();
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

	float samplePoint(hlp::fvec2 offset) {
		hlp::fvec2 origin = {
				(float)(actor.position.x + display.getScreenWidth() / 2),
				(float)(-actor.position.y + display.getScreenHeight() / 2)
		};
		hlp::fvec2 samplePos = origin + offset;
		samplePos = hlp::fvec2::rotatePoint(samplePos, origin, -actor.rotation);

		float value = 0.0f;
		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++) {
				if (samplePos.x + x >= 0 && samplePos.x + x < display.getScreenWidth() &&
					samplePos.y + y >= 0 && samplePos.y + y < display.getScreenHeight()) {
					value += display.getPixel(samplePos.x + x, samplePos.y + y).r;
					display.drawPixel(samplePos.x + x, samplePos.y + y, hlp::color(255, 0, 0));
				}
			}
		}

		return value / (9.0f * 255.0f);
	}
};
