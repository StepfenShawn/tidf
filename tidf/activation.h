#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <cmath>

#include "matrix.h"

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
}

#endif /* _ACTIVATION_H_  */