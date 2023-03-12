#ifndef _NET_H_
#define _NET_H_

#include "matrix.h"
#include "activation.h"
#include "loss.h"
#include "layer.h"

class Net {
    private:
        std::vector<Layer> Layers;
        std::vector<int> caches;
    public:
        Net(std::vector<Layer> Layers);
        Net();
};

#endif /* _NET_H_ */