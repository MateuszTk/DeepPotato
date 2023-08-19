#pragma once
#include <initializer_list>

// #define CHECK_INDEX

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

	template <typename... Args>
	const T* dataAt(Args... args) const {
		return data + getIndex(args...);
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

