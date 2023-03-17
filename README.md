# Tidf
A tiny but fast deep-learning framework in c++.  

# Why Tidf?
* Powerful: `Tidf` provides a Keras-like api.
* Simple and stupid: Support brandcast and made in pure c++11.
* Tiny but fast: The implementation is under 1,000 semicolons.

# Components
#### Layers
* Dense Layer (Linear Layer)

#### Activations
* sigmoid
* relu
* tanh

#### Optimizers
* SGD (Stochastic gradient descent)

### Cast Functions
* CrossEntropy
* MSE (L2Loss)
* L1Loss

more to come! :)

# Examples

### Train a 3-Layers neural network
```cpp
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
    net->addLayer(LayerType::Dense, 4, "tanh");
    net->addLayer(LayerType::Dense, 1, "sigmoid");
    net->compile("CrossEntropyLoss", "SGD");
    net->fit(train_inputs, train_outputs, 50000);
    std::cout << net->predict(
      MAT( double, ({{1.0}, {1.0}, {1.0}})) ) << std::endl;
    return 0;
}
```
Result:  
```
Matrix(1 x 1):
0.998889
```

### Matrix
`tidf` supports `brandcast` and provides a simple but powerful `API` to make it easier to work with matrices:  
```cpp
#include "tidf/matrix.h"

int main() {
    NEW_MAT(m1, double, ({{1, 4, 3}, {1, 2, 3}}));
    NEW_MAT(m2, double, ({{1, 2, 3}}));
    NEW_MAT(m3, double, ({{2, 3, 4}}));
    std::cout << m1.dot(m2.transpose()) << std::endl;
    std::cout << m2 * 2.0 + m3 << std::endl;
    std::cout << m2.apply([](double x) -> double {return x * 10.0;}) << std::endl;
    std::cout << m2.join(m3) << std::endl;
    std::cout << m1.row(1).sum() << std::endl;
    return 0;
}
```
# License
MIT License  
Copyright (c) 2023 Stepfen Shawn