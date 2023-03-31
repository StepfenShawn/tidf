#ifndef _LOSS_H_
#define _LOSS_H_

#include <assert.h>

namespace Loss {
    template <class T>
    Matrix<T> L1Loss(Matrix<T> x, Matrix<T> y) {
        T N = y.col_size;
        Matrix<T> sum(x.row_size, 1);
        sum.fill((T)0);
        for (int i = 0; i < N; i++) {
            sum = sum + (x.col(i) - y.col(i)).apply( [](T x) -> T { return fabs(x); } );
        }
        return sum / N;
    }

    template <class T>
    Matrix<T> L1LossBackward(Matrix<T> x, Matrix<T> y) { return Matrix<T>(); }

    template <class T>
    Matrix<T> MSELoss(Matrix<T> x, Matrix<T> y) { return Matrix<T>(); }

    template <class T>
    Matrix<T> MSELossBackward(Matrix<T> x, Matrix<T> y) { return Matrix<T>(); }

    template <class T>
    Matrix<T> CrossEntropyLoss(Matrix<T> x, Matrix<T> y) {
        T N = y.col_size;
        std::function<T(T)> f_log = [](T x) -> T { return log(x); };
        std::function<T(T)> f1 = [](T x) -> T { return (T)1 - x; };
        Matrix<T> cast = y * x.apply(f_log) + (y.apply(f1) * x.apply(f1).apply(f_log));
        return cast * (-1 / N);
    }

    template <class T>
    Matrix<T> CrossEntropyLossBackward(Matrix<T> predict, Matrix<T> outputs) {
        return ( -outputs ) / predict
             + outputs.apply([](T x) -> T { return (T)1 - x; }) /
               predict.apply([](T x) -> T { return (T)1 - x; });
    }
};

#endif /* _LOSS_H_ */