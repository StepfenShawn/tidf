#ifndef _LAYER_H_
#define _LAYER_H_

enum class LayerType {
    Dense = 0,
    Conval   
};

template <class T>
class Layer {
    private:
        Matrix<T> linear_forward_activation(Matrix<T> Z);

        // -------------- Dropout Part ---------------
        bool use_dropout = false;
        T keep_prob;
        Matrix<T> DL; // Mask Matrix to shut down the some neurons
        // -------------------------------------------

    public:
        // numbers of the neurons
        int dims;
        LayerType type;
        std::string activation;
        Matrix<T> weights;
        Matrix<T> bias;

        Matrix<T> dw;
        Matrix<T> db;
        Matrix<T> da;
        Matrix<T> Z;
        Matrix<T> dZ;

        Matrix<T> input;
        Matrix<T> output;

        // Creates layer.
        Layer(int dims);
        Layer(LayerType type, int dims);
        Layer(LayerType type, int dims, std::string string);

        // -------------------- Init -------------------------
        void initWeight(int row_size, int col_size);
        void initBias(int row_size, int col_size);
        void setInput(Matrix<T> input)   { this->input = input; }
        void setOutput(Matrix<T> output) { this->output = output; }
        // ---------------------------------------------------
        
        void useDropout(const T keep_prob) { 
            this->use_dropout = true;
            this->keep_prob = keep_prob; 
        }

        Matrix<T> linear_forward();

        // backward: dLoss->dA->dZ->dw, db
        void linear_backward(Matrix<T> dZ, Matrix<T> A_prev, T m);
        void linear_backward_activation(Matrix<T> dA, Matrix<T> A_prev, T m);

        std::string __str__();
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
    this->weights = this->weights.to_random();
}

template <class T>
void Layer<T>::initBias(int row_size, int col_size) {
    Matrix<T> mat(row_size, col_size);
    this->bias = mat;
    this->bias.fill((T)0);
}

// Calculate the "AL" on current layer.
template <class T>
Matrix<T> Layer<T>::linear_forward() {
    switch (this->type)
    {
        case LayerType::Dense:
            this->Z = this->weights.dot(this->input) + this->bias;
            break;
        
        default:
            break;
    }
    Matrix<T> AL;
    if (this->use_dropout) {
        int col_size = this->Z.col_size;
        int row_size = this->Z.row_size;
        randmat(b, T, (row_size, col_size));
        this->DL = b < this->keep_prob;
        // Shut down some neurons of A[l]
        AL = this->linear_forward_activation(this->Z) * DL;
        // Scale the value of neurons that haven't been shut down
        AL = AL / this->keep_prob;
    }
    else
        AL = this->linear_forward_activation(this->Z);
    return AL;
}

template <class T>
Matrix<T> Layer<T>::linear_forward_activation(Matrix<T> Z) {
    if (this->activation == "sigmoid")
        return Activation::sigmoid(Z);
    else if (this->activation == "relu)                                                                                                                                                                                                                                                   ")
        return Activation::relu(Z);
    else if (this->activation == "tanh")
        return Activation::_tanh(Z);
    else
        return Z;
}

template <class T>
void Layer<T>::linear_backward(Matrix<T> dZ, Matrix<T> A_prev, T m) {
    this->dw = this->dZ.dot(A_prev.transpose()) * (1 / m);
    this->db = Matrix<T>(dZ.row_size, 1);
    this->db.fill((T)0);
    for (int i = 0; i < this->dZ.col_size; i++)
        this->db = this->db + dZ.col(i);
    this->db = this->db * (1 / m);
}

// notice: the parameter dA come from last layer
template <class T>
void Layer<T>::linear_backward_activation(Matrix<T> dA, Matrix<T> Activation_cache, T m) {

    if (this->use_dropout) {
        // Apply mask Matrix to shut down the same neurons as during the forward proganation.
        dA = dA * this->DL;
        // Scale the value of neurons that haven't been shut down
        dA = dA / keep_prob;
    }

    if (this->activation == "sigmoid")
        this->dZ = dA * Activation::deriv_sigmoid(Activation_cache);
    else if (this->activation == "relu")
        this->dZ = dA * Activation::deriv_relu(Activation_cache);
    else if (this->activation == "tanh")
        this->dZ = dA * Activation::deriv_tanh(Activation_cache);

    this->linear_backward(this->dZ, this->input, m);
}

template <class T>
std::string Layer<T>::__str__() {
    return "Neurons: " + std::to_string(this->dims) + "\n" +   
           "Type: " + "(" + std::to_string((int)this->type) + ")\n";
}

#endif /* _lAYER_H_ */