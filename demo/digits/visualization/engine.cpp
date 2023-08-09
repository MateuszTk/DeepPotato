#include "engine.h"

//--------------IO--------------//

SDL_Surface* engine::IO::openTexture(const char* path) {
	SDL_Surface* surface = SDL_LoadBMP(path);
	if (surface == NULL) {
		printf("Failed to load: %s error: %s\n", path, SDL_GetError());
		return nullptr;
	};
	return surface;
}

const Uint8* engine::IO::getKeys(SDL_Event* event, bool* quit) {
	while (SDL_PollEvent(event)) {
		//quit game
		if (event->type == SDL_QUIT)
			*quit = true;
	}
	return SDL_GetKeyboardState(NULL);
}

//--------------Display--------------//

engine::Display::Display(int _screenWidth, int _screenHeight, const char* windowTitle, bool initFont) : screenWidth(_screenWidth), screenHeight(_screenHeight) {
	if (initializeView(windowTitle) && initFont) {
		initializeFont();
	}
}

engine::Display::~Display() {
	SDL_FreeSurface(this->screen);
	SDL_FreeSurface(this->charset);
	SDL_DestroyTexture(this->sceenTexture);
	SDL_DestroyWindow(this->window);
	SDL_DestroyRenderer(this->renderer);
	SDL_Quit();
}

double engine::Display::printFrame(bool clear) {
	SDL_UpdateTexture(this->sceenTexture, NULL, this->screen->pixels, this->screen->pitch);
	SDL_RenderCopy(this->renderer, this->sceenTexture, NULL, NULL);
	SDL_RenderPresent(this->renderer);

	if (clear)
		SDL_FillRect(this->screen, NULL, SDL_MapRGB(this->screen->format, 0x00, 0x00, 0x00));

	//calculate delta time
	int t2 = SDL_GetTicks();
	float deltaTime = (t2 - this->time1) * 0.001;
	this->time1 = t2;

	//calculate fps
	this->fpsTimer += deltaTime;
	if (fpsTimer > 0.5) {
		this->fps = frames * 2;
		this->frames = 0;
		this->fpsTimer -= 0.5;
	};
	this->frames++;

	return deltaTime;
}

void engine::Display::drawString(const char* text, hlp::vec2<int> position, float fontSize, bool background, bool centered) {
	if (centered) position.x -= (int)strlen(text) * 8 * fontSize / 2;
	SDL_SetColorKey(this->charset, !background, 0x000000);
	int c;
	SDL_Rect source, destination;
	source.w = 8;
	source.h = 8;
	destination.w = source.w * fontSize;
	destination.h = source.h * fontSize;

	while (*text) {
		c = *text & 255;
		//position on the source chrset bitmap
		source.x = (c % 16) * 8;
		source.y = (c / 16) * 8;
		//position on the screen
		destination.x = position.x;
		destination.y = position.y;
		//print character
		SDL_BlitScaled(this->charset, &source, this->screen, &destination);
		//move cursor
		position.x += destination.w;
		text++;
	};
}

void engine::Display::drawPixel(int x, int y, hlp::color color) {
	Uint8* pixels = (Uint8*)this->screen->pixels;
	const int bpp = this->screen->format->BytesPerPixel;
	const int id = (x + y * this->screenWidth) * bpp;
	pixels[id + 2] = color.r;
	pixels[id + 1] = color.g;
	pixels[id] = color.b;
}

hlp::color engine::Display::getPixel(int x, int y){
	Uint8* pixels = (Uint8*)this->screen->pixels;
	const int bpp = this->screen->format->BytesPerPixel;
	const int id = (x + y * this->screenWidth) * bpp;
	return {pixels[id + 2],
		pixels[id + 1],
		pixels[id],
	};
}

void engine::Display::drawHLine(hlp::vec2<int> start, int length, hlp::color color) {
	start = toScreenCenter(start);
	for (int y = 0; y < length; y++) {
		drawPixel(start.x, start.y + y, color);
	}
}

void engine::Display::drawSurface(SDL_Surface* surface, hlp::vec2<int> pos) {
	SDL_Rect dest;
	dest.x = pos.x - surface->w / 2;
	dest.y = pos.y - surface->h / 2;
	dest.w = surface->w;
	dest.h = surface->h;
	SDL_BlitSurface(surface, NULL, this->screen, &dest);
}

int engine::Display::getScreenWidth() {
	return this->screenWidth;
}

int engine::Display::getScreenHeight() {
	return this->screenHeight;
}

int engine::Display::getFPS() {
	return this->fps;
}

hlp::vec2<int> engine::Display::toScreenCenter(hlp::vec2<int>& position) {
	return (position + hlp::vec2<int>(screenWidth / 2, screenHeight / 2));
}


bool engine::Display::initializeFont() {
	//open font image
	this->charset = IO::openTexture(CHARSET_PATH);

	if (this->charset) {
		//key alpha
		SDL_SetColorKey(this->charset, true, 0x000000);
		return true;
	}
	return false;
}

bool engine::Display::initializeView(const char* windowTitle) {
	if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
		printf("SDL_Init error: %s\n", SDL_GetError());
		return false;
	}

	// 0 - ok; -1 - failed
	int windowOk = SDL_CreateWindowAndRenderer(this->screenWidth, this->screenHeight, 0, &this->window, &this->renderer);
	if (windowOk != 0) {
		SDL_Quit();
		printf("SDL_CreateWindowAndRenderer error: %s\n", SDL_GetError());
		return false;
	};

	SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");
	SDL_RenderSetLogicalSize(this->renderer, this->screenWidth, this->screenHeight);
	SDL_SetRenderDrawColor(this->renderer, 0, 0, 0, 255);

	this->screen = SDL_CreateRGBSurface(0, this->screenWidth, this->screenHeight, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);

	this->sceenTexture = SDL_CreateTexture(
		this->renderer,
		SDL_PIXELFORMAT_ARGB8888,
		SDL_TEXTUREACCESS_STREAMING,
		this->screenWidth,
		this->screenHeight
	);

	SDL_SetWindowTitle(this->window, windowTitle);

	//hide cursor
	SDL_ShowCursor(SDL_DISABLE);

	return true;
}

//--------------Actor--------------//

bool engine::Actor::fitsX(hlp::rect<double> rect) {
	return (position.x - surface->w / 2 >= rect.x - rect.w / 2 && position.x + surface->w / 2 <= rect.x + rect.w / 2);
}

bool engine::Actor::fitsY(hlp::rect<double> rect) {
	return (position.y - surface->h / 2 >= rect.y - rect.h / 2 && position.y + surface->h / 2 <= rect.y + rect.h / 2);
}

bool engine::Actor::fits(hlp::rect<double> rect) {
	return fitsX(rect) && fitsY(rect);
}

bool engine::Actor::isInsideX(double pointX) {
	return (pointX <= position.x + surface->w / 2 && pointX >= position.x - surface->w / 2);
}

bool engine::Actor::isInside(hlp::vec2<double> point) {
	return (isInsideX(point.x) && point.y <= position.y + surface->h / 2 && point.y >= position.y - surface->h / 2);
}

bool engine::Actor::isInsideRect(hlp::vec2<double> point, hlp::rect<double> rect) {
	return (point.x <= rect.x + rect.w / 2 && point.x >= rect.x - rect.w / 2 &&
		point.y <= rect.y + rect.h / 2 && point.y >= rect.y - rect.h / 2);
}

bool engine::Actor::overlaps(hlp::rect<double> rect) {
	return (
		isInside({ rect.x + rect.w / 2, rect.y + rect.h / 2 }) ||
		isInside({ rect.x + rect.w / 2, rect.y - rect.h / 2 }) ||
		isInside({ rect.x - rect.w / 2, rect.y + rect.h / 2 }) ||
		isInside({ rect.x - rect.w / 2, rect.y - rect.h / 2 })
		);
}

//check if part of the actor area is in the box 
bool engine::Actor::overlapsRect(hlp::rect<double> rect) {
	return (
		isInsideRect({ position.x + surface->w / 2, position.y + surface->h / 2 }, rect) ||
		isInsideRect({ position.x + surface->w / 2, position.y - surface->h / 2 }, rect) ||
		isInsideRect({ position.x - surface->w / 2, position.y + surface->h / 2 }, rect) ||
		isInsideRect({ position.x - surface->w / 2, position.y - surface->h / 2 }, rect)
		);
}

bool engine::Actor::collides(hlp::rect<double> rect) {
	return overlaps(rect) || overlapsRect(rect);
}

//--------------Scene--------------//

void engine::Scene::addActor(Actor* actor) {
	actors.push_back(actor);
}

void engine::Scene::drawActors(Display* display) {
	for (int i = 0; i < actors.getSize(); i++) {
		if (actors[i]->surface && actors[i]->active) {
			hlp::vec2<int> posOnScreen = actors[i]->position - this->cameraPosition;
			posOnScreen.y *= -1;
			display->drawSurface(actors[i]->surface, display->toScreenCenter(posOnScreen));
		}
	}
}

engine::Actor* engine::Scene::findActor(const char* name) {
	for (int i = 0; i < actors.getSize(); i++) {
		Actor* actor = actors[i];
		if (strcmp(actor->name, name) == 0) return actor;
	}
	return nullptr;
}

void engine::Scene::setCameraPosition(hlp::vec2<double> newPosition) {
	this->cameraPosition = newPosition;
}

hlp::vec2<double> engine::Scene::getCameraPosition() {
	return this->cameraPosition;
}

