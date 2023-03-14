#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

// A series of functions to update weights.
namespace Optimizer {
    
    // Stochastic gradient descent.
    template <class T>
    Matrix<T> SGD(Matrix<T>& weights, Matrix<T>& dweights, double learning_rate) {
        weights = weights - dweights * (T)learning_rate;
        return weights;
    }
};

#endif /* _OPTIMIZER_H_  */