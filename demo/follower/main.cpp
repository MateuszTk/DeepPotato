
// ------------------- CONFIG -------------------
	// --------TRAINING--------
		// number of threads used for training, set to 0 or not define to disable multithreading
			#define THREAD_POOL_SIZE 4

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

#define SDL_MAIN_HANDLED
#include "engine.h"
#include "Canvas.hpp"

#include "LineFollower.hpp"

int main(int argc, char** argv) {
	SDL_Event event;
	bool quit = false;
	double deltaTime = 0.0;
	double time = 0.0;
	double lastTime = 0.0;

	LineFollower follower;

	while (true) {
		const Uint8* keys = engine::IO::getKeys(&event, &quit);
		if (quit) break;
		if (keys[SDL_SCANCODE_SPACE]) follower.reset();
		if (keys[SDL_SCANCODE_D]) follower.action(0, 1.0 * deltaTime);
		if (keys[SDL_SCANCODE_A]) follower.action(1, 1.0 * deltaTime);
		if (keys[SDL_SCANCODE_W]) follower.action(2, 40.0 * deltaTime);
		if (keys[SDL_SCANCODE_S]) follower.action(3, 40.0 * deltaTime);

		follower.updateCanvas();

		Matrix1D<float> state = follower.sampleLine();
		std::cout << std::fixed << std::setprecision(2);
		for (int i = 0; i < state.getDimension(0); i++) {
			std::cout << state(i) << ' ';
		}
		std::cout << std::endl;

		if (time - lastTime >= 0.1) {
			if (state(1) > 0.0) follower.action(1, 0.1);
			if (state(state.getDimension(0) - 2) > 0.0) follower.action(0, 0.1);
			lastTime = time;
		}
		follower.action(2, 40.0 * deltaTime);

		deltaTime = follower.print();
		time += deltaTime;
	}
	
	return 0;
}
