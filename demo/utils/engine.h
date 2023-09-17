#pragma once
#define CHARSET_PATH ""
#include "helper.h"

namespace engine {
	namespace IO {
		SDL_Surface* openTexture(const char* path);
		const Uint8* getKeys(SDL_Event* event, bool* quit);
	}

	class Display {
	public:
		Display(int _screenWidth, int _screenHeight, const char* windowTitle, bool initFont = true);
		~Display();
		double printFrame(bool clear = true);
		void drawString(const char* text, hlp::vec2<int> position, float fontSize = 1.0f, bool background = false, bool centered = false);
		void drawPixel(int x, int y, hlp::color color);
		void drawPixel(int x, int y, hlp::color color, SDL_Surface* surface);
		void drawHLine(hlp::vec2<int> start, int length, hlp::color color);
		void drawSurface(SDL_Surface* surface, hlp::vec2<int> pos, float rotation = 0.0f);
		hlp::color getPixel(int x, int y);
		hlp::color getPixel(int x, int y, SDL_Surface* surface);
		int getScreenWidth();
		int getScreenHeight();
		int getFPS();
		hlp::vec2<int> toScreenCenter(hlp::vec2<int>& position);

	private:
		const int screenWidth;
		const int screenHeight;

		int time1 = 0, fps = 0;
		double fpsTimer = 0.0;
		int frames = 0;

		//collections of actual pixels
		SDL_Surface* screen = nullptr;
		SDL_Surface* charset = nullptr;

		SDL_Texture* sceenTexture = nullptr;
		SDL_Window* window = nullptr;
		SDL_Renderer* renderer = nullptr;

		SDL_Surface* rotateSurface(SDL_Surface* surface, float angle);

		bool initializeFont();
		bool initializeView(const char* windowTitle);	
	};

	struct Actor {
		SDL_Surface* surface;
		hlp::vec2<double> position;
		float rotation;
		const char* name;
		bool active = true; 
		hlp::vec2<double> momentum;

		//check if the actor is entirely in the rect area (X axis check only)
		bool fitsX(hlp::rect<double> rect);
		//check if the actor is entirely in the rect area (Y axis check only)
		bool fitsY(hlp::rect<double> rect);
		//check if the actor is entirely in the rect area
		bool fits(hlp::rect<double> rect);
		//check if point is in the actor area (check only X axis)
		bool isInsideX(double pointX);
		//check if point is in the actor area
		bool isInside(hlp::vec2<double> point);
		//check if point is in the rect area
		bool isInsideRect(hlp::vec2<double> point, hlp::rect<double> rect);
		//check if part of the rect is in the actor area
		bool overlaps(hlp::rect<double> rect);
		//check if part of the actor area is in the box 
		bool overlapsRect(hlp::rect<double> rect);
		//check using both overlaps and overlapsRect methods
		bool collides(hlp::rect<double> rect);
		//move actor by delta in local coordinates
		void move(hlp::vec2<double> delta);
	};

	class Scene {
	public:
		void addActor(Actor* actor);
		void drawActors(Display* display);
		Actor* findActor(const char* name);
		void setCameraPosition(hlp::vec2<double> newPosition);
		hlp::vec2<double> getCameraPosition();

	private:
		hlp::vector<Actor*> actors;
		hlp::vec2<double> cameraPosition;
	};
}
