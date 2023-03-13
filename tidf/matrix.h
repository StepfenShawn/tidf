#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <functional>
#include <ctime>
#include <cstdlib>

#define randint(a, b) (rand() % (b - a) + a)
#define random() (rand() / double(RAND_MAX))

#define _RANDOM_INIT_ \
    srand((int)time(NULL))

#define NEW_MAT(name, type, value) \
        Matrix<type> name = Matrix<type>(std::vector<std::vector<type > > value )

#define NEW_MAT_SIZE(name, type, shape) \
        Matrix<type> name = Matrix<type> shape

#define MAT_BLOCK(m, startRow, startCol, row_size, col_size) \
        ( m.Matblock(startRow, startCol, row_size, col_size) )

#define DOT(m1, m2) \
        ( m1.dot(m2) )

#define TRANSPOSE(m) \
        ( m.transpose() )

template <class T>
class Matrix {
    private:
        // we don't need cols
        // because cols = (rows.T).rows
        std::vector<std::vector<T> > rows;
        Matrix<T> brandcast(Matrix<T> const& m) const;

    public:
        std::vector<std::vector<T> > mat_arr;
        int row_size;
        int col_size;
        // Constructors
        Matrix<T>(int row_size, int col_size);
        Matrix<T>(std::vector<std::vector<T> > const& mat_arr);
        Matrix<T>();

        int getSizeOfRow() const { return this->row_size; }
        int getSizeOfCol() const { return this->col_size; }
        T get(int h, int w) const;

        Matrix<T> col(size_t i) const;
        Matrix<T> row(size_t i) const;

        Matrix<T> addCol(Matrix<T>& newCol);
        Matrix<T> addRow(Matrix<T>& newRow);

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

        Matrix<T> join(const Matrix<T>& m) const;

        bool sameShape(const Matrix<T>& m) const;
        T sum() const;

        Matrix<T> to_ramdom();        
        void fill(T value);
};

template <class T> Matrix<T> operator + (const Matrix<T>& a, const Matrix<T>& b);
template <class T> Matrix<T> operator - (const Matrix<T>& a, const Matrix<T>& b);
template <class T> Matrix<T> operator * (T a, const Matrix<T>& m);
template <class T> Matrix<T> operator * (const Matrix<T>& m, T a);

template <class T> Matrix<T> operator + (T a, Matrix<T>& m);
template <class T> Matrix<T> operator + (Matrix<T>& m, T a);

template <class T> std::ostream& operator << (std::ostream& flux, const Matrix<T>& m);
template <class T> std::wostream& operator << (std::wostream& flux, const Matrix<T>& m);

// connect 2 Matrixs
template <class T> Matrix<T> operator >> (const Matrix<T>& a, const Matrix<T>& b);

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

    this->rows = this->mat_arr;
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

template <class T>
Matrix<T> Matrix<T>::col(size_t i) const {
    NEW_MAT(result, T, { this->transpose().mat_arr[i] });
    return result;
}

template <class T>
Matrix<T> Matrix<T>::row(size_t i) const {
    NEW_MAT(result, T, { this->mat_arr[i] });
    return result;
}

template <class T>
T Matrix<T>::get(int h, int w) const {
    if (!(h >= 0 && h < this->row_size && w > 0 && w < this->col_size)) {
        throw std::invalid_argument("Index out of bounds. ");
    }
    return this->mat_arr[h][w];
}

template <class T>
void Matrix<T>::fill(T value) {
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            this->mat_arr[i][j] = value;
        }
    }
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
    if (!this->sameShape(m)){
        if (this->col_size == m.col_size || this->row_size == m.row_size) {
            // if (this->col_size )
        } else {
            throw std::invalid_argument(" ValueError: frames are not aligned! ");
        }
    }
    Matrix<T> m1 = m.brandcast(*this);
    
    if (!this->sameShape(m1))
        throw std::invalid_argument(" Error: catched in brandcasting! ");

    Matrix<T> result(this->row_size, this->col_size);
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            result.mat_arr[i][j] = this->mat_arr[i][j] + m1.mat_arr[i][j];
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
Matrix<T> Matrix<T>::join(const Matrix<T>& m) const {
    if (!(this->col_size == m.col_size))
        throw std::invalid_argument("2 matrixs' col_size are not same. ");
    std::vector<std::vector<T> > result_mat_arr{ this->mat_arr };
    for (auto e : m.mat_arr){
        result_mat_arr.push_back(e);
    }
    Matrix<T> result(result_mat_arr);
    return result;
}

template <class T>
Matrix<T> Matrix<T>::addRow(Matrix<T>& newRow) {
    return this->join(newRow);
}

template <class T>
Matrix<T> Matrix<T>::addCol(Matrix<T>& newCol) {
    return this->transpose().join(newCol).transpose();
}

template <class T>
Matrix<T> Matrix<T>::brandcast(Matrix<T> const& m) const {
    Matrix<T> result = *this;
    if (this->col_size == m.col_size) {
        Matrix<T> lastRow = this->row(this->row_size - 1);
        for (int ii = 0; ii < m.row_size - this->row_size; ii++) {
            result.join(lastRow);
        }
    } else if (this->row_size == m.row_size) {
        Matrix<T> lastCol = this->col(this->col_size - 1);
        for (int ii = 0; ii < m.col_size - this->col_size; ii++) {
            result = result.transpose().join(lastCol).transpose();
        }
    }
    return result;
}

template <class T>
T Matrix<T>::sum() const {
    T sum = 0;
    for (int i = 0; i < this->row_size; i++) {
        for (int j = 0; j < this->col_size; j++) {
            sum += this->mat_arr[i][j];
        }
    }
    return sum;
}

template <class T>
Matrix<T> Matrix<T>::to_ramdom() {
    return this->apply([](T x) -> T { return random(); });
}

template <class T>
Matrix<T> operator + (const Matrix<T>& a, const Matrix<T>& b) {
    return a.add(b);
}

template <class T>
Matrix<T> operator + (const Matrix<T>& m, T a) {
    return m.apply([a](T x) -> T {return x + a;});
}

template <class T>
Matrix<T> operator + (T a, const Matrix<T>& m) {
    return m.apply([a](T x) -> T {return x + a;});
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

template <class T>
Matrix<T> operator >> (const Matrix<T>& a, const Matrix<T>& b) {
    return a.join(b);
}