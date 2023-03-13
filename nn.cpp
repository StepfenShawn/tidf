#include "tidf/net.h"

int main() {
    _RANDOM_INIT_;
    // 4 * 3
    NEW_MAT(train_inputs, double, 
      ({{0.0, 0.0, 1.0},
        {1.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}}));

    NEW_MAT(train_outputs, double,
      ({{0.0},
        {1.0},
        {1.0},
        {0.0}}));

    Net<double>* net = new Net<double>(train_inputs, train_outputs);
    net->addLayer(LayerType::Dense, 3, "sigmoid");
    net->addLayer(LayerType::Dense, 5, "sigmoid");
    net->initParams();
    net->train(train_inputs, train_outputs, 10000);
    std::cout << net->getLayer(2).output << std::endl;
    return 0;
}