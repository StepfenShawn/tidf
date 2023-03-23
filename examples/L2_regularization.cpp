#include "../tidf/core.h"

int main() {
    Matrix<double> train_input = load_mat<double>("train_X.txt");
    Matrix<double> train_ouput = load_mat<double>("train_Y.txt");

    Net<double>* net = new Net<double>(train_input, train_ouput);
    net->addLayer(LayerType::Dense, 3, "sigmoid");
    net->Dropout();
    net->addLayer(LayerType::Dense, 3, "tanh");
    net->addLayer(LayerType::Dense, 1, "sigmoid");
    net->compile("CrossEntropyLoss", "SGD");
    net->fit(train_input, train_input, 1000);
    std::cout << (net->predict(train_input) - train_ouput).sum() << std::endl;
    return 0;
}