#ifndef _NET_H_
#define _NET_H_

#include "layer.h"
#include "optimizer.h"

#define _TIDF_INIT_ \
    _RANDOM_INIT_

template <class T>
class Net {
    private:
        std::vector<Layer<T> > Layers;
        Matrix<T> inputs;
        Matrix<T> outputs;

        std::string loss;
        std::string optimizer;

        // Type: Matrix<T> (predict), Matrix<T> (output) -> Matrix<T>
        std::function<Matrix<T>(Matrix<T>, Matrix<T>)> f_loss;
        std::function<Matrix<T>(Matrix<T>, Matrix<T>)> f_loss_backward;

        void thinking(Matrix<T> inputs, Matrix<T> outputs);

    public:
        // Constructs
        Net();
        Net(Matrix<T> inputs, Matrix<T> outputs);

        void initParams();
        void addLayer(LayerType type, int dim, std::string activation);
        void fit(Matrix<T> inputs, Matrix<T> outputs, int iters);
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
void Net<T>::thinking(Matrix<T> inputs, Matrix<T> outputs) {
    int num_layers = this->Layers.size();
    // linear forward
    for (int L = 1; L < num_layers; L++) {
        this->Layers[L].setInput(this->Layers[L - 1].output);
        this->Layers[L].output = this->Layers[L].linear_forward();
    }
    Matrix<T> predict = this->Layers[num_layers - 1].output;
    Matrix<T> AL = this->f_loss_backward(predict, this->outputs);
    Matrix<T> cast = this->f_loss(predict, this->outputs);

    // std::cout << cast.sum() << std::endl;
    // backward propagation
    for (int L = this->Layers.size() - 1; L >= 1; L--) {
        if (L == this->Layers.size() - 1)
            this->Layers[L].linear_backward_activation(AL, this->Layers[L].Z);
        else
            this->Layers[L].linear_backward_activation(this->Layers[L].da, this->Layers[L].Z);
        this->Layers[L - 1].da = this->Layers[L].weights.transpose().dot(this->Layers[L].dZ);
    
        this->Layers[L].weights = this->Layers[L].weights - 0.05 * this->Layers[L].dw;
        this->Layers[L].bias = this->Layers[L].bias - 0.05 * this->Layers[L].db;
    }
}

template <class T>
void Net<T>::fit(Matrix<T> inputs, Matrix<T> outputs, int iters) {
    for (int iter = 0; iter < iters; iter++)
        this->thinking(inputs, outputs);
}

template <class T>
Layer<T> Net<T>::getLayer(int index) {
    return this->Layers[index];
}

template <class T>
void Net<T>::compile(std::string loss, std::string optimizer) {
    this->initParams();
    this->loss = loss;
    
    if (this->loss == "CrossEntropyLoss") {
        this->f_loss = Loss::CrossEntropyLoss<T>;
        this->f_loss_backward = Loss::CrossEntropyLossBackward<T>;
    } else if (this->loss == "L1Loss") {
        this->f_loss = Loss::L1Loss<T>;
        this->f_loss_backward = Loss::L1LossBackward<T>;
    } else if (this->loss == "MSELoss" || this->loss == "L2Loss") {
        this->f_loss = Loss::MSELoss<T>;
        this->f_loss_backward = Loss::MSELossBackward<T>;
    } else {
        throw std::invalid_argument("Unknown cast funtion: " + loss);
    }

    this->optimizer = optimizer;
}

template <class T>
Matrix<T> Net<T>::predict(Matrix<T> inputs) {
    int num_layers = this->Layers.size();
    this->Layers[0].setOutput(inputs);
    // linear forward
    for (int L = 1; L < num_layers; L++) {
        this->Layers[L].setInput(this->Layers[L - 1].output);
        this->Layers[L].output = this->Layers[L].linear_forward();
    }
    Matrix<T> predict = this->Layers[num_layers - 1].output;
    return predict;
}

#endif /* _NET_H_ */