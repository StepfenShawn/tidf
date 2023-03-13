#include "tidf/matrix.h"

int main() {   
    NEW_MAT(m1, double, ({{1, 4, 3}, {1, 2, 3}}));
    NEW_MAT(m2, double, ({{1, 2, 3}}));
    NEW_MAT(m3, double, ({{2, 3, 4}}));
    NEW_MAT(m4, double, ({{1}}));
    std::cout << m1.dot(m2.transpose()) << std::endl;
    std::cout << m2 * 2.0 + m3 << std::endl;
    std::cout << m2.apply([](double x) -> double {return x * 10.0;}) << std::endl;
    std::cout << m2.join(m3) << std::endl;
    std::cout << m1.row(1).sum() << std::endl;
    _RANDOM_INIT_;
    std::cout << m1.to_ramdom() << std::endl;
    // brandcast:
    std::cout << m3 + m4 << std::endl;
    return 0;
}