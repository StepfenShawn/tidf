# Tidf
[Wip] A tiny but fast deep-learning framework in c++.  

# Examples

### Matrix
```cpp
#include "tidf/matrix.h"

int main() {
    NEW_MAT(m1, double, ({{1, 4, 3}, {1, 2, 3}}));
    NEW_MAT(m2, double, ({{1, 2, 3}}));
    NEW_MAT(m3, double, ({{2, 3, 4}}));
    std::cout << m1.dot(m2.transpose()) << std::endl;
    std::cout << m2 * 2.0 + m3 << std::endl;
    std::cout << m2.apply([](double x) -> double {return x * 10.0;}) << std::endl;
    return 0;
}
```

### Activation function
```cpp
#include "tidf/activation.h"

int main() {
    NEW_MAT(m1, double, ({{10.1, 10.1, 10.1}}));
    std::cout << Activation::sigmoid(m1.transpose()) << std::endl;
    std::cout << Activation::deriv_sigmoid(m1.transpose()) << std::endl;
    return 0;
}
```