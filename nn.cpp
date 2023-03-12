#include "tidf/net.h"

int main() {
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

    Net* net = new Net();
    // net->addLayer(Dense(3, "sigmoid"));
    // net->train(train_inputs, train_outputs, 10000);
    // std::cout << net->predict(Matrix) << std::endl;
    return 0;
}