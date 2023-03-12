#include "tidf/activation.h"

int main() {
    std::cout << Activation::sigmoid(10.1) << std::endl;
    std::cout << Activation::deriv_sigmoid(10.1) << std::endl;
    NEW_MAT(m1, double, ({{10.1, 10.1, 10.1}}));
    std::cout << Activation::sigmoid(m1.transpose()) << std::endl;
    std::cout << Activation::deriv_sigmoid(m1.transpose()) << std::endl;
    return 0;
}