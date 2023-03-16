#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

// A series of functions to update weights.
namespace Optimizer {
    
    // Stochastic gradient descent.
    template <class T>
    void SGD(Matrix<T>& weights, Matrix<T>& dw, 
                  Matrix<T>& bias, Matrix<T>& db, double learning_rate) {
        weights = weights - dw * learning_rate;
        bias = bias - db * learning_rate;
    }

    template <class T>
    void SGD(Matrix<T>& weights, Matrix<T>& dw, double learning_rate) {
        weights = weights - dw * learning_rate;
    }
};

#endif /* _OPTIMIZER_H_  */