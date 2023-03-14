#ifndef _LAYER_H_
#define _LAYER_H_

#include "matrix.h"
#include "activation.h"
#include "loss.h"

enum class LayerType {
    Dense = 0,
    Conval   
};

template <class T>
class Layer {
    private:
        Matrix<T> linear_forward_activation(Matrix<T> Z);
    public:
        int dims;
        LayerType type;
        std::string activation;
        Matrix<T> weights;
        Matrix<T> bias;

        Matrix<T> dw;
        Matrix<T> db;

        Matrix<T> input;
        Matrix<T> output;

        // Creates layer.
        Layer(int dims);
        Layer(LayerType type, int dims);
        Layer(LayerType type, int dims, std::string string);

        void initWeight(int row_size, int col_size);
        void initBias(int row_size, int col_size);

        void setInput(Matrix<T> input)   { this->input = input; }
        void setOutput(Matrix<T> output) { this->output = output; }

        Matrix<T> linear_forward();
};

template <class T>
Layer<T>::Layer(int dims) {
    this->dims = dims;
}

template <class T>
Layer<T>::Layer(LayerType type, int dims) {
    this->dims = dims;
}

template <class T>
Layer<T>::Layer(LayerType type, int dims, std::string activation) {
    this->type = type;
    this->dims = dims;
    this->activation = activation;
}

template <class T>
void Layer<T>::initWeight(int row_size, int col_size) {
    Matrix<T> mat(row_size, col_size);
    this->weights = mat;
    this->weights = this->weights.to_ramdom();
}

template <class T>
void Layer<T>::initBias(int row_size, int col_size) {
    Matrix<T> mat(row_size, col_size);
    this->bias = mat;
    this->bias = this->bias.to_ramdom();
}

template <class T>
Matrix<T> Layer<T>::linear_forward() {
    Matrix<T> Z;

    switch (this->type)
    {
        case LayerType::Dense:
            Z = this->weights.dot(this->input) + this->bias;
            break;
        
        default:
            break;
    }
    return this->linear_forward_activation(Z);
}

template <class T>
Matrix<T> Layer<T>::linear_forward_activation(Matrix<T> Z) {
    if (this->activation == "sigmoid")
        return Activation::sigmoid(Z);
    else if (this->activation == "relu)                                                                                                                                                                                                                                                   ")
        return Activation::relu(Z);
    else
        return Z;
}

#endif /* _lAYER_H_ */