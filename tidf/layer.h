#ifndef _LAYER_H_
#define _LAYER_H_

#include "matrix.h"

class Layer {
    public:
        // Creates layer.
        Layer();
};

template <class T>
class Dense : public Layer {
    public:
        Matrix<T> inputs;
        Matrix<T>* weights;
        Matrix<T>* bias;

        Dense(int units, std::string activation);
        Dense(int units, std::string activation, bool use_bias);


    private:
        std::string activation;
        bool use_bias;
        Matrix<T> linear_forward();
        Matrix<T> backward();
};

template <class T>
Dense<T>::Dense<T>(int units, std::string activation) {
    this->weights = new Matrix<T>(units, 1);
    this->bias = new Matrix<T>(units, 1);
    this->activation = activation;
}

template <class T>
Dense<T>::Dense<T>(int units, std::string activation, bool use_bias) {
    this->activation = this->activation;
    if (use_bias) this(units, activation);
    else {
        this->use_bias = false;
        this->weights = new Matrix<T>(units, 1);
    }
}

template <class T>
Matrix<T> Dense<T>::linear_forward() {
    Matrix<T> Z;
    Matrix<T> A;
    if (this->use_bias)
        Z = this->weights.dot(this->inputs) + this->bias;
    else
        Z = this->weights.dot(this->inputs);

    switch (this->activation)
    {
        case "sigmoid":
            A = Activation::sigmoid(Z);
            break;
    }

    return A;
}

template <class T>
Matrix<T> Dense<T>::backward() {
    
}

#endif /* _lAYER_H_ */