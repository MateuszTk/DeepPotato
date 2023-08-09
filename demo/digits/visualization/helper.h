#define _CRT_SECURE_NO_WARNINGS 
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" {
#include "SDL.h"
#include "SDL_main.h"
}

//typedef unsigned char byte;

namespace hlp {
	//check if any of the arguments is equal
	template<typename T, typename... Args>
	bool anyEqual(T first, Args... comp) {
		//array of the results of the comparisons
		bool compareResults[] = { (first == comp)... };
		int size = sizeof(compareResults) / sizeof(compareResults[0]);
		for (int i = 0; i < size; i++) {
			if (compareResults[i]) return true;
		}
		//all comparisons were false, so return false
		return false;
	}

	//return smaller value
	template <typename T>
	inline T min(T a, T b) {
		return ((a < b) ? a : b);
	}

	//return bigger value
	template <typename T>
	inline T max(T a, T b) {
		return ((a > b) ? a : b);
	}

	//rise to the power of two
	template <typename T>
	inline T pow2(T a) {
		return a * a;
	}

	//check if value is in range (inclusive)
	template <typename T>
	inline bool isInRangeIn(T value, T min, T max) {
		return (value >= min && value <= max);
	}

	//check if value is in range (exclusive)
	template <typename T>
	inline bool isInRangeEx(T value, T min, T max) {
		return (value > min && value < max);
	}

	//clamp value
	template <typename T>
	inline T clamp(T v, T minv, T maxv) {
		return ((v > maxv) ? maxv : ((v < minv) ? minv : v));
	}

	//return sign of the value
	template <typename T>
	inline int sign(T v) {
		return ((v >= 0) ? 1 : -1);
	}

	template <typename T>
	class vector {
	public:
		vector() {
			data = nullptr;
			size = 0;
			capacity = 0;
		}

		~vector() {
			if (data != nullptr)
				delete[] data;
		}

		void push_back(T value) {
			//allocate more memory if neeed
			if (size >= capacity) {
				capacity = (capacity == 0) ? 4 : capacity * 2;
				T* newData = new T[capacity];
				for (int i = 0; i < size; i++) {
					newData[i] = data[i];
				}
				delete[] data;
				data = newData;
			}
			//add an element to the end of the vector
			data[size] = value;
			size++;
		}

		T& operator[](int index) {
			return data[index];
		}

		void clear() {
			size = 0;
			delete[] data;
			capacity = 0;
			data = nullptr;
		}

		int getSize() {
			return this->size;
		}

		T& getLast() {
			return this->data[max(this->size - 1, 0)];
		}

		T* getData() {
			return this->data;
		}

	private:
		T* data;
		int size;
		int capacity;
	};

	struct color {
		unsigned char r;
		unsigned char g;
		unsigned char b;
		unsigned char a;

		color(unsigned char _r, unsigned char _g, unsigned char _b, unsigned char _a = 255) : r(_r), g(_g), b(_b), a(_a) {

		}

		color() {
			r = g = b = 0;
			a = 255;
		}
	};

	template <typename T>
	struct rect {
		T x, y, w, h;
	};

	template <typename T>
	struct vec2 {
		T x, y;
		vec2(T _x, T _y) : x(_x), y(_y) {}
		vec2() : x(0), y(0) {}

		template <typename U>
		vec2(const vec2<U>& second) {
			this->x = second.x;
			this->y = second.y;
		}

		vec2 operator+(const vec2& second) const {
			return vec2(this->x + second.x, this->y + second.y);
		}
		vec2& operator+=(const vec2& second) {
			this->x += second.x;
			this->y += second.y;
			return *this;
		}
		vec2 operator-(const vec2& second) const {
			return vec2(this->x - second.x, this->y - second.y);
		}
		vec2& operator-=(const vec2& second) {
			this->x -= second.x;
			this->y -= second.y;
			return *this;
		}
		vec2 operator*(const T scalar) const {
			return vec2(this->x * scalar, this->y * scalar);
		}
		vec2 operator/(const T scalar) const {
			return vec2(this->x / scalar, this->y / scalar);
		}
		bool operator==(const vec2& second) const {
			return (this->x == second.x && this->y == second.y);
		}
	};

	template <typename A, typename B>
	struct pair {
		A first;
		B second;
		pair(A _first, B _second) : first(_first), second(_second) {

		}
		bool operator==(const pair& other) const {
			return (this->first == other->first && this->second == other->second);
		}
	};
}
