#include "tidf/net.h"

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
    net->addLayer(LayerType::Dense, 4, "sigmoid");
    net->addLayer(LayerType::Dense, 4, "sigmoid");
    net->addLayer(LayerType::Dense, 1, "sigmoid");
    net->compile("CrossEntropyLoss", "SGD");
    net->fit(train_inputs, train_outputs, 50000);
    std::cout << net->predict(
      MAT( double, ({{1.0}, {1.0}, {1.0}})) ) << std::endl;
    return 0;
}