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
    net->addLayer< 4 >(LayerType::Dense, "sigmoid");
    net->Dropout(0.84);
    net->addLayer< 4 >(LayerType::Dense, "tanh"); 
    net->addLayer< 1 >(LayerType::Dense, "sigmoid");
    net->compile("CrossEntropyLoss", "SGD");
    // Show the config of the neural network
    std::cout << net->get_config() << std::endl;
    net->fit(100000);
    std::cout << net->predict(
      MAT( double, ({{1.0}, {1.0}, {1.0}})) ) << std::endl;
    return 0;
}