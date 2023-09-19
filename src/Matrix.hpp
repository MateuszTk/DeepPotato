#pragma once
#include <initializer_list>
#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>

// #define CHECK_INDEX

template <typename T, unsigned int nDim>
class Matrix {
public:
	Matrix(const std::array<unsigned int, nDim>& dimensions) : isSubMatrix(false) {
		this->size = 1;
		for (unsigned int i = 0; i < nDim; ++i) {
			this->dimensions[i] = dimensions[i];
			this->size *= dimensions[i];
		}
		if (this->size > 0) {
			this->data = new T[this->size];
		}
		else {
			this->data = nullptr;
		}
	}

	Matrix(const Matrix<T, nDim + 1>& master, unsigned int index) : isSubMatrix(true) {
		this->size = 1;
		for (unsigned int i = 0; i < nDim; ++i) {
			this->dimensions[i] = master.getDimension(i);
			this->size *= master.getDimension(i);
		}

		if (this->size > 0) {
			this->data = master.getData() + index;
		}
		else {
			this->data = nullptr;
		}
	}

	Matrix(const Matrix& other) {
		this->size = 1;
		for (unsigned int i = 0; i < nDim; ++i) {
			this->dimensions[i] = other.dimensions[i];
			this->size *= this->dimensions[i];
		}
		this->data = new T[this->size];
		*this = other;
	}

	Matrix& operator=(const Matrix& other) {
		for (unsigned int i = 0; i < size; ++i) {
			data[i] = other.data[i];
		}
		return *this;
	}

	template <typename... Args>
	decltype(auto) operator()(Args... args) const {
		constexpr unsigned int argSize = sizeof...(Args);
		const int index = getIndex(args...);
		if constexpr (argSize >= nDim) {
			return (T&)(data[index]);
		}
		else {
			return std::make_unique<Matrix<T, nDim - argSize>>(*this, index);
		}
	}

	const T* begin() const {
		return data;
	}

	const T* end() const {
		return data + size;
	}

	T* getData() const {
		return data;
	}

	unsigned int getDimension(unsigned int dim) const {
		return dimensions[dim];
	}

	void setAll(T value) {
		for (unsigned int i = 0; i < size; ++i) {
			data[i] = value;
		}
	}

	template <typename... Args>
	T* dataAt(Args... args) const {
		return data + getIndex(args...);
	}

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

	const Matrix& add(const Matrix& other) {
		if (this->size != other.size) {
			throw std::invalid_argument("Matrix sizes do not match");
		}
		for (unsigned int i = 0; i < this->size; i++) {
			data[i] += other.data[i];
		}
		return *this;
	}

	const Matrix& subtract(const Matrix& other) {
		if (this->size != other.size) {
			throw std::invalid_argument("Matrix sizes do not match");
		}
		for (unsigned int i = 0; i < this->size; i++) {
			data[i] -= other.data[i];
		}
		return *this;
	}
	
	const Matrix& operator+=(const Matrix& other) {
		return add(other);
	}

	const Matrix& operator-=(const Matrix& other) {
		return subtract(other);
	}

	const Matrix& operator*=(T scalar) {
		for (unsigned int i = 0; i < this->size; i++) {
			data[i] *= scalar;
		}
		return *this;
	}

	~Matrix() {
		if (!isSubMatrix) {
			delete[] data;
		}
	}

private:
	T* data;
	unsigned int size;
	unsigned int dimensions[nDim];
	bool isSubMatrix;

	template <typename... Args>
	inline const auto getIndex(Args... args) const {
		const unsigned int argsArr[] = { args... };
		constexpr unsigned int argSize = sizeof...(Args);
		constexpr unsigned int argNDim = (argSize > nDim) ? nDim : argSize;

#ifdef CHECK_INDEX
		for (unsigned int i = 0; i < argNDim; i++) {
			if (argsArr[i] >= dimensions[i]) {
				throw std::out_of_range("Index out of range");
			}
		}
#endif
		
		unsigned int index = 0;
		unsigned int multi = 1;

		constexpr unsigned int argDiff = nDim - argNDim;
		if constexpr (argSize < nDim) {
			for (unsigned int i = 0; i < argDiff; i++) {
				multi *= dimensions[i];
			}
		}

		for (unsigned int i = 0; i < argNDim; i++) {
			index += argsArr[i] * multi;
			multi *= dimensions[i + argDiff];
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
	unsigned int aCols = a.getDimension(0);
	unsigned int aRows = a.getDimension(1);

	unsigned int bRows = b.getDimension(0);

	unsigned int cRows = c.getDimension(0);

	unsigned int resultRows = result.getDimension(0);

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
	else {
		std::cout << "Matrix test_0 passed" << std::endl;
	}

	Matrix3D<float> m({ 2, 2, 2 });
	m(0, 0, 0) = 1;
	m(0, 0, 1) = 2;
	m(0, 1, 0) = 3;
	m(0, 1, 1) = 4;
	m(1, 0, 0) = 5;
	m(1, 0, 1) = 6;
	m(1, 1, 0) = 7;
	m(1, 1, 1) = 8;

	const int z = 1;
	std::cout << m(0, 0, z) << " " << m(0, 1, z) << " " << m(1, 0, z) << " " << m(1, 1, z) << std::endl;

	auto m2 = m(z);
	std::cout << (*m2)(0, 0) << " " << (*m2)(0, 1) << " " << (*m2)(1, 0) << " " << (*m2)(1, 1) << std::endl;

	if ((*m2)(0, 0) != 2 || (*m2)(0, 1) != 4 || (*m2)(1, 0) != 6 || (*m2)(1, 1) != 8) {
		throw std::runtime_error("Matrix test failed");
	}
	else {
		std::cout << "Matrix test_1 passed" << std::endl;
	}
}

