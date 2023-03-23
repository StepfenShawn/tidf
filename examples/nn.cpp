#include "../tidf/core.h"

int main() {
    _TIDF_INIT_;
    NEW_MAT(train_inputs, double, 
      ({{0.0, 0.0, 1.0},
        {1.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}}));

    NEW_MAT(train_outputs, double,
      ({{0.0, 1.0, 1.0, 0.0}}));
    
    Net<double>* net = new Net<double>(train_inputs.transpose(), train_outputs);
    net->addLayer<4>(LayerType::Dense, "sigmoid"); 
    net->addLayer<4>(LayerType::Dense, "tanh"); 
    net->addLayer<1>(LayerType::Dense, "sigmoid");
    net->compile("CrossEntropyLoss", "SGD");
    net->fit(100000);
    std::cout << net->predict(
      MAT( double, ({{1.0}, {1.0}, {1.0}})) ) << std::endl;
    return 0;
}