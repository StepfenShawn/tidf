#include "../tidf/core.h"

int main() {
    Matrix<double> train_X = load_mat<double>("train_X.txt");
    Matrix<double> train_Y = load_mat<double>("train_Y.txt");
    
    std::cout << train_X << std::endl;
    std::cout << train_Y << std::endl;
}