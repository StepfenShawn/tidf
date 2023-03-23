#ifndef _NET_H_
#define _NET_H_

#define _TIDF_INIT_ \
    _RANDOM_INIT_

template <class T>
class Net {
    private:
        std::vector<Layer<T> > Layers;
        Matrix<T> inputs;
        Matrix<T> outputs;

        // -------------- Layers -----------------------
        int curLayerIndex = -1;
        Layer<T> getCurLayer();
        // ---------------------------------------------

        std::string loss;
        std::string optimizer;
        // -------------- Regularization ----------------
        std::string regularization = "";
        double lambda;
        double keep_prob;

        Matrix<T> L2Regularization(T m, double lambda, Matrix<T> weights);

        // ----------------------------------------------
        
        double learning_rate = 0.05;

        // Type: Matrix<T> (predict), Matrix<T> (output) -> Matrix<T>
        std::function<Matrix<T>(Matrix<T>, Matrix<T>)> f_loss;
        std::function<Matrix<T>(Matrix<T>, Matrix<T>)> f_loss_backward;

        void thinking(Matrix<T> inputs, Matrix<T> outputs);

    public:
        // Constructs
        Net() {};
        Net(Matrix<T> inputs, Matrix<T> outputs);

        void initParams();
        template <int dim>
        void addLayer(LayerType type, std::string activation);
        
        // ---------- Regularization ---------------
        void Dropout(double keep_prob);
        void SetL2Regularization(double lambda);
        void SetL1Regularization(double lambda);
        // -----------------------------------------

        void fit(long long iters);
        Matrix<T> predict(Matrix<T> inputs);

        void compile(std::string loss, std::string optimizer);
        void compile(std::function<Matrix<T>(Matrix<T>, Matrix<T>)> f_loss, 
                    std::function<Matrix<T>(Matrix<T>, Matrix<T>)> f_loss_back,
                    std::string optimizer);

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
template <int dim>
void Net<T>::addLayer(LayerType type, std::string activation) {
    this->Layers.push_back(Layer<T>(type, dim, activation));
}

template <class T>
void Net<T>::Dropout(double keep_prob) {
    Matrix<T> D1;
    int col_size = this->getCurLayer().output.col_size;
    int row_size = this->getCurLayer().output.row_size;
    Matrix<T> b(row_size, col_size);
    b.to_ramdom();
    D1 = b.apply([&](T x) -> T { return x < keep_prob ? 1 : 0; });
}

template <class T>
void Net<T>::SetL2Regularization(double lambda) {
    this->regularization = "L2";
    this->lambda = lambda;
}

template <class T>
Matrix<T> Net<T>::L2Regularization(T m, double lambda, Matrix<T> weights) {
    return weights * (lambda / m);
}

template <class T>
void Net<T>::thinking(Matrix<T> inputs, Matrix<T> outputs) {
    int num_layers = this->Layers.size();
    T m = this->inputs.col_size;
    // linear forward
    for (int L = 1; L < num_layers; L++) {
        this->curLayerIndex = L;
        this->Layers[L].setInput(this->Layers[L - 1].output);
        this->Layers[L].output = this->Layers[L].linear_forward();
    }
    Matrix<T> predict = this->Layers[num_layers - 1].output;
    Matrix<T> AL = this->f_loss_backward(predict, this->outputs);
    T cast = this->f_loss(predict, this->outputs).sum();
    if (this->regularization == "L2") {
        for (int L = 1; L < num_layers; L++)
            cast = cast + (this->Layers[L].weights * this->Layers[L].weights).sum();
    }

    std::cout << cast << std::endl;
    // backward propagation
    for (int L = this->Layers.size() - 1; L >= 1; L--) {
        this->curLayerIndex = L;
        if (L == this->Layers.size() - 1)
            this->Layers[L].linear_backward_activation(AL, this->Layers[L].Z, m);
        else
            this->Layers[L].linear_backward_activation(this->Layers[L].da, this->Layers[L].Z, m);
        this->Layers[L - 1].da = this->Layers[L].weights.transpose().dot(this->Layers[L].dZ);

        if (this->regularization == "L2") {
            this->Layers[L].dw = this->Layers[L].dw + this->L2Regularization(m, this->lambda, this->Layers[L].weights);
        }
        if (this->optimizer == "SGD")
            Optimizer::SGD(this->Layers[L].weights, this->Layers[L].dw, 
                        this->Layers[L].bias, this->Layers[L].db, this->learning_rate);
    }
}

template <class T>
void Net<T>::fit(long long iters) {
    for (long long iter = 0; iter < iters; iter++)
        this->thinking(this->inputs, this->outputs);
}

template <class T>
Layer<T> Net<T>::getLayer(int index) {
    return this->Layers[index];
}

template <class T>
Layer<T> Net<T>::getCurLayer() {
    return this->Layers[this->curLayerIndex];
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
void Net<T>::compile(std::function<Matrix<T>(Matrix<T>, Matrix<T>)> f_loss,
                     std::function<Matrix<T>(Matrix<T>, Matrix<T>)> f_loss_backward,
                     std::string optimizer) {
    this->f_loss = f_loss;
    this->f_loss_backward = f_loss_backward;
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

template <class T>
std::string Net<T>::get_config() const {
    return "Loss: " + this->loss + "\n" + 
            "Optitimizer: " + this->optimizer + "\n";
}

#endif /* _NET_H_ */