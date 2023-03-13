#ifndef _NET_H_
#define _NET_H_

#include "layer.h"

template <class T>
class Net {
    private:
        std::vector<Layer<T>> Layers;
        Matrix<T> inputs;
        Matrix<T> outputs;
    public:
        // Constructs
        Net();
        Net(Matrix<T> inputs, Matrix<T> outputs);

        void initParams();
        void addLayer(LayerType type, int dim, std::string activation);
        void train(Matrix<T> inputs, Matrix<T> outputs, int iters);
        Layer<T> getLayer(int index);

        std::string get_config() const;
};

template <class T>
Net<T>::Net(Matrix<T> inputs, Matrix<T> outputs) {
    this->inputs = inputs;
    this->outputs = outputs;
    Layer<T> input_layer(this->inputs.row_size);
    input_layer.setOutput(inputs);
    this->Layers.push_back(input_layer);
}

template <class T>
void Net<T>::initParams() {
    for (int L = 1; L < this->Layers.size(); L++) {
        this->Layers[L].initWeight(this->Layers[L].dims, this->Layers[L - 1].dims);
        this->Layers[L].initBias(this->Layers[L].dims, 1);
    }
}

template <class T>
void Net<T>::addLayer(LayerType type, int dim, std::string activation) {
    this->Layers.push_back(Layer<T>(type, dim));
}

template <class T>
void Net<T>::train(Matrix<T> inputs, Matrix<T> outputs, int iters) {
    for (int L = 1; L < this->Layers.size(); L++) {
        this->Layers[L].setInput(this->Layers[L - 1].output);
        this->Layers[L].output = this->Layers[L].linear_forward();
    }
}

template <class T>
Layer<T> Net<T>::getLayer(int index) {
    return this->Layers[index];
}

#endif /* _NET_H_ */