#include "../tidf/core.h"

int main() {
    Matrix<double> train_input = load_mat<double>("train_X.txt");
    Matrix<double> train_ouput = load_mat<double>("train_Y.txt");

    Net<double>* net = new Net<double>(train_input, train_ouput);
    net->addLayer<3>(LayerType::Dense, "sigmoid");
    net->addLayer<3>(LayerType::Dense, "tanh");
    net->addLayer<1>(LayerType::Dense, "sigmoid");
    net->compile("CrossEntropyLoss", "SGD");
    net->fit(3000);
    std::cout << (net->predict(train_input) - train_ouput).apply([](double x) -> double {return fabs(x); }).sum() << std::endl;
    return 0;
}