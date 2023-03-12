#ifndef _LOSS_H_
#define _LOSS_H_

#include "matrix.h"

#include <assert.h>

namespace Loss {
    template <class T>
    Matrix<T> L1Loss(Matrix<T> x, Matrix<T> y) {
        assert(x.sameShape(y));
        int N = x.col_size;
        Matrix<T> sum(x.row_size, 1);
        sum.fill((T)0);
        for (int i = 0; i < N; i++) {
            sum = sum + (x.col(i) - y.col(i));
        }
        return sum / N;
    }

    template <class T>
    Matrix<T> MSELoss(Matrix<T> x, Matrix<T> y) {
        return;
    }

    template <class T>
    Matrix<T> CrossEntropyLoss(Matrix<T> x, Matrix<T> y) {
        return;
    }
}

#endif /* _LOSS_H_ */