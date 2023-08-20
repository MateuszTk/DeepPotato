#pragma once
#include <initializer_list>
#include <array>
#include <functional>
#include <iostream>

//#define CHECK_INDEX

template <typename T, unsigned int nDim>
class Matrix {
public:
	Matrix(const std::array<unsigned int, nDim>& dimensions) {
		this->size = 1;
		for (unsigned int i = 0; i < nDim; ++i) {
			this->dimensions[i] = dimensions[i];
			this->size *= dimensions[i];
		}
		if (size > 0) {
			this->data = new T[size];
		}
		else {
			this->data = nullptr;
		}
		this->size = size;
	}

	Matrix(const Matrix& other) {
		*this = other;
	}

	Matrix& operator=(const Matrix& other) {
		for (unsigned int i = 0; i < size; ++i) {
			data[i] = other.data[i];
		}
		return *this;
	}

	template <typename... Args>
	T& operator()(Args... args) {
		int index = getIndex(args...);
		return data[index];
	}

	template <typename... Args>
	const T& operator()(Args... args) const {
		int index = getIndex(args...);
		return data[index];
	}

	const T* begin() const {
		return data;
	}

	const T* end() const {
		return data + size;
	}

	const T* getData() const {
		return data;
	}

	unsigned int getDimension(unsigned int dim) const {
		return dimensions[dim];
	}

	template <typename... Args>
	T* dataAt(Args... args) const {
		return data + getIndex(args...);
	}

	template<typename T>
	friend void multiplyAndAdd(const Matrix<T, 2>& a, const Matrix<T, 1>& b, const Matrix<T, 1>& c, Matrix<T, 1>& result);

	void applyFunction(std::function<T(T)> function) {
		for (unsigned int i = 0; i < this->size; i++) {
			this->data[i] = function(this->data[i]);
		}
	}

	void applyFunction(const Matrix& source, std::function<T(T)> function) {
		for (unsigned int i = 0; i < this->size; i++) {
			this->data[i] = function(source.data[i]);
		}
	}

	~Matrix() {
		delete[] data;
	}

private:
	T* data;
	unsigned int size;
	unsigned int dimensions[nDim];

	template <typename... Args>
	inline const unsigned int getIndex(Args... args) const {
		const unsigned int argsArr[] = { args... };

#ifdef CHECK_INDEX
		for (unsigned int i = 0; i < nDim; i++) {
			if (argsArr[i] >= dimensions[i]) {
				throw std::out_of_range("Index out of range");
			}
		}
#endif
		
		unsigned int index = 0;
		unsigned int multi = 1;

		for (unsigned int i = 0; i < nDim; i++) {
			index += argsArr[i] * multi;
			multi *= dimensions[i];
		}
		return index;
	}
};

template <typename T>
using Matrix1D = Matrix<T, 1>;

template <typename T>
using Matrix2D = Matrix<T, 2>;

template <typename T>
using Matrix3D = Matrix<T, 3>;

// result = a * b + c
template <typename T>
void multiplyAndAdd(const Matrix2D<T>& a, const Matrix1D<T>& b, const Matrix1D<T>& c, Matrix1D<T>& result) {
	unsigned int aCols = a.dimensions[0];
	unsigned int aRows = a.dimensions[1];

	unsigned int bRows = b.dimensions[0];

	unsigned int cRows = c.dimensions[0];

	unsigned int resultRows = result.dimensions[0];

	if (aCols == bRows && aRows == cRows && aRows == resultRows) {
		for (unsigned int i = 0; i < aRows; i++) {
			T sum = c(i);
			T* aRowStart = a.dataAt(0, i);
			T* bRowStart = b.dataAt(0);
			for (unsigned int j = 0; j < aCols; j++) {
				sum += aRowStart[j] * bRowStart[j];
			}
			result(i) = sum;
		}
	}
	else {
		throw std::invalid_argument("Invalid matrix dimensions");
	}
}

void testMatrix() {
	Matrix2D<float> a({ 3, 2 });
	Matrix1D<float> b({ 3 });
	Matrix1D<float> c({ 2 });
	Matrix1D<float> result({ 2 });

	a(0, 0) = 1;
	a(1, 0) = 2;
	a(2, 0) = 3;
	a(0, 1) = 4;
	a(1, 1) = 5;
	a(2, 1) = 6;

	b(0) = 7;
	b(1) = 8;
	b(2) = 9;

	c(0) = 1;
	c(1) = 2;

	multiplyAndAdd(a, b, c, result);

	for (unsigned int i = 0; i < result.getDimension(0); i++) {
		std::cout << result(i) << std::endl;
	}

	if (result(0) != 51 || result(1) != 124) {
		throw std::runtime_error("Matrix test failed");
	}
}

