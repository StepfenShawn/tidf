#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <cmath>

namespace Activation {
    
    template <class T>
    T sigmoid(T x) {
        return 1 / ( 1 + exp(-x) );
    }
    
    template <class T>
    Matrix<T> sigmoid(const Matrix<T> m) {
        std::function<T(T)> f_sigmoid = [](T x) -> T { return sigmoid(x); };
        return m.apply(f_sigmoid); 
    }

    template <class T>
    T deriv_sigmoid(T x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    template <class T>
    Matrix<T> deriv_sigmoid(const Matrix<T> m) {
        std::function<T(T)> f_deriv_sigmoid = [](T x) -> T { return deriv_sigmoid(x); };
        return m.apply(f_deriv_sigmoid);
    }

    template <class T>
    T relu(T x) {
        return std::max((T)0, x);
    }

    template <class T>
    Matrix<T> relu(const Matrix<T> m) {
        std::function<T(T)> f_relu = [](T x) -> T { return relu(x); };
        return m.apply(f_relu);
    }

    template <class T>
    T deriv_relu(T x) {
        return x > 0 ? 1 : 0;
    }

    template <class T>
    Matrix<T> deriv_relu(const Matrix<T> m) {
        std::function<T(T)> f_deriv_relu = [](T x) -> T { return deriv_relu(x); };
        return m.apply(f_deriv_relu);
    }

};

#endif /* _ACTIVATION_H_  */