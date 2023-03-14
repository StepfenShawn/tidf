#ifndef _NET_H_
#define _NET_H_

#include "layer.h"
#include "optimizer.h"

template <class T>
class Net {
    private:
        std::vector<Layer<T> > Layers;
        Matrix<T> inputs;
        Matrix<T> outputs;

        std::string loss;
        std::string optimizer;

    public:
        // Constructs
        Net();
        Net(Matrix<T> inputs, Matrix<T> outputs);

        void initParams();
        void addLayer(LayerType type, int dim, std::string activation);
        void train(Matrix<T> inputs, Matrix<T> outputs, int iters);
        Matrix<T> predict(Matrix<T> inputs);

        void compile(std::string loss, std::string optimizer);

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
    this->Layers.push_back(Layer<T>(type, dim, activation));
}

template <class T>
void Net<T>::train(Matrix<T> inputs, Matrix<T> outputs, int iters) {
    int num_layers = this->Layers.size();
    // linear forward
    for (int L = 1; L < num_layers; L++) {
        this->Layers[L].setInput(this->Layers[L - 1].output);
        this->Layers[L].output = this->Layers[L].linear_forward();
    }
    Matrix<T> predict = this->Layers[num_layers - 1].output;
    Matrix<T> cast = Loss::L1Loss(predict, this->outputs);
    // calcalaute loss
    // this->Layers[num_layers - 1] = Loss::L1Loss();
    // backward pagation        
    for (int L = this->Layers.size() - 1; L > 0; L--) {

    }
}

template <class T>
Layer<T> Net<T>::getLayer(int index) {
    return this->Layers[index];
}

template <class T>
void Net<T>::compile(std::string loss, std::string optimizer) {
    this->loss = loss;
    this->optimizer = optimizer;
}

template <class T>
Matrix<T> Net<T>::predict(Matrix<T> inputs) {
    return Matrix<T>();
}

#endif /* _NET_H_ */