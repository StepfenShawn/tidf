#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <functional>

#define NEW_MAT(name, type, value) \
        Matrix<type> name = Matrix<type>(std::vector<std::vector<type > > value )

#define MAT_BLOCK(m, startRow, startCol, row_size, col_size) \
        ( m.Matblock(startRow, startCol, row_size, col_size) )

#define DOT(m1, m2) \
        ( m1.dot(m2) )

#define TRANSPOSE(m) \
        ( m.transpose() )

template <class T>
class Matrix {
    private:
        std::vector<std::vector<T> > mat_arr;
        int row_size;
        int col_size;

    public:
        // Constructors
        Matrix<T>(int row_size, int col_size);
        Matrix<T>(std::vector<std::vector<T> > const& mat_arr);
        Matrix<T>();

        int getSizeOfRow() const { return this->row_size; }
        int getSizeOfCol() const { return this->col_size; }
        T get(int h, int w) const;

        // Apply function to each element in Matrix
        Matrix<T> apply(T (*function)(T)) const;
        Matrix<T> apply(std::function<T(T)>& f) const;

        // Let's make a sub-Matrix
        Matrix<T> Matblock(size_t startH, size_t startW, size_t row_size, size_t col_size) const;

        void __str__(std::ostream& flux) const;
        void __str__(std::wostream& flux) const;

        Matrix<T> add(const Matrix<T>& m) const;
        Matrix<T> sub(const Matrix<T>& m) const;
        // element-wise operation
        Matrix<T> mul(const Matrix<T>& m) const;
        // Scalar
        Matrix<T> mul(const T& value) const;

        Matrix<T> dot(const Matrix<T>& m) const;
        Matrix<T> divide(const T& value) const;
        // Get the transpose of Matrix
        Matrix<T> transpose() const;

        bool sameShape(const Matrix<T>& m) const;
};

template <class T> Matrix<T> operator + (const Matrix<T>& a, const Matrix<T>& b);
template <class T> Matrix<T> operator - (const Matrix<T>& a, const Matrix<T>& b);
template <class T> Matrix<T> operator * (T a, const Matrix<T>& m);
template <class T> Matrix<T> operator * (const Matrix<T>& m, T a);

// TODO: brandcast
template <class T> Matrix<T> operator + (T a, Matrix<T>& m);

template <class T> std::ostream& operator << (std::ostream& flux, const Matrix<T>& m);
template <class T> std::wostream& operator << (std::wostream& flux, const Matrix<T>& m);

#endif /* _MATRIX_H_ */

template <class T>
Matrix<T>::Matrix(int row_size, int col_size) {
    this->row_size = row_size;
    this->col_size = col_size;
    this->mat_arr = std::vector<std::vector<T> >(row_size, std::vector<T>(col_size));
}

template <class T>
Matrix<T>::Matrix(std::vector<std::vector<T> > const &mat_arr) {
    if (mat_arr.size() == 0) {
        throw std::invalid_argument("Size of mat_arr must greater than 0");
    }

    this->row_size = mat_arr.size();
    this->col_size = mat_arr[0].size();
    this->mat_arr = mat_arr;
}

template <class T>
Matrix<T>::Matrix() {
    this->row_size = 0;
    this->col_size = 0;
}

template<class T>
void Matrix<T>::__str__(std::ostream& flux) const {
    flux << "Matrix(" << this->row_size << " x " << this->col_size  << "):" << std::endl;
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            flux << this->mat_arr[i][j] << " ";
        }
        flux << std::endl;
    }
}

template<class T>
void Matrix<T>::__str__(std::wostream& flux) const {
    flux << L"Matrix(" << this->row_size << " x " << this->col_size  << "):" << std::endl;
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            flux << this->mat_arr[i][j] << L" ";
        }
        flux << std::endl;
    }
}

template<class T>
std::ostream& operator << (std::ostream& flux, const Matrix<T>& m) {
    m.__str__(flux);
    return flux;
}

template<class T>
std::wostream& operator << (std::wostream& flux, const Matrix<T>& m) {
    m.__str__(flux);
    return flux;
}

template<class T>
T Matrix<T>::get(int h, int w) const {
    if (!(h >= 0 && h < this->row_size && w > 0 && w < this->col_size)) {
        throw std::invalid_argument("Index out of bounds. ");
    }
    return this->mat_arr[h][w];
}

template <class T>
Matrix<T> Matrix<T>::apply(T (*function)(T)) const {
    Matrix<T> result(this->row_size, this->col_size);

    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            result.mat_arr[i][j] = (*function)(this->mat_arr[i][j]);
        }
    }
    return result;
}

template <class T>
Matrix<T> Matrix<T>::apply(std::function<T(T)>& f) const {
    Matrix<T> result(this->row_size, this->col_size);

    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            result.mat_arr[i][j] = f(this->mat_arr[i][j]);
        }
    }
    return result;
}

template <class T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(this->col_size, this->row_size);

    for (int i = 0; i < this->col_size; i++) {
        for (int j = 0; j < this->row_size; j++) {
            result.mat_arr[i][j] = this->mat_arr[j][i];
        }
    }

    return result;
}

template <class T>
Matrix<T> Matrix<T>::Matblock(size_t startH, size_t startW, size_t row_size, size_t col_size) const {
    if (!(startH >= 0 && startH + row_size <= this->row_size 
       && startW >= 0 && startW + col_size <= this->col_size)) {
        throw std::invalid_argument("Index out of bounds.");
    }
    Matrix<T> result(row_size, col_size);
    for (int i = startH; i < startH + row_size; i++) {
        for (int j = startW; j < startW + col_size; j++) {
            result.mat_arr[i - startH][j - startW] = this->mat_arr[i][j];
        }
    }

    return result;
}

template <class T>
bool Matrix<T>::sameShape(const Matrix<T>& m) const {
    return (this->row_size == m.row_size && this->col_size == m.col_size);
}

template <class T>
Matrix<T> Matrix<T>::add(const Matrix<T>& m) const {
    if (!this->sameShape(m)) 
        throw std::invalid_argument(" Matrix dimension must be the same!  ");
    Matrix<T> result(this->row_size, this->col_size);
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            result.mat_arr[i][j] = this->mat_arr[i][j] + m.mat_arr[i][j];
        }
    }
    return result;
}

template <class T>
Matrix<T> Matrix<T>::sub(const Matrix<T>& m) const {
    if (!this->sameShape(m)) 
        throw std::invalid_argument(" Matrix dimension must be the same!  ");
    Matrix<T> result(this->row_size, this->col_size);
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            result.mat_arr[i][j] = this->mat_arr[i][j] - m->mat_arr[i][j];
        }
    }
    return result;
}

template <class T>
Matrix<T> Matrix<T>::mul(const T& value) const {
    std::function<T(T)> f = [value](T x) -> T { return x * value; };
    return this->apply(f);
}

template <class T>
Matrix<T> Matrix<T>::divide(const T& value) const {
    std::function<T(T)> f = [value](T x) -> T { return x / value; };
    return this->apply(f);
}

template <class T>
Matrix<T> Matrix<T>::dot(const Matrix<T>& m) const {
    if (!(this->col_size == m.row_size))
        throw std::invalid_argument("Dot product not compatible.");

    Matrix<T> result(this->row_size, m.col_size);

    for (int i = 0; i < row_size; i++) {
        for (int j = 0; j < m.col_size; j++) {
            T w = (T)0;
            for (int h = 0; h < this->col_size; h++) {
                w += this->mat_arr[i][h] * m.mat_arr[h][j];
            }
            result.mat_arr[i][j] = w;
            w = (T)0;
        }
    }

    return result;
}

template <class T>
Matrix<T> operator + (const Matrix<T>& a, const Matrix<T>& b) {
    return a.add(b);
}

template <class T>
Matrix<T> operator - (const Matrix<T>& a, const Matrix<T>& b) {
    return a.sub(b);
}

template <class T>
Matrix<T> operator * (const Matrix<T>& m, T a) {
    return m.mul(a);
}

template <class T>
Matrix<T> operator * (T a, const Matrix<T>& m) {
    return m.mul(a);
}