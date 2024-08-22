#include "optimizer.cpp"
#include "option_cleaner.cpp"
#include "option_cleaner.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

PYBIND11_MODULE(bindings, m) {
    m.doc() = ""; 

    m.def("performOptimization", &performOptimization, "");
    
    m.def("getFeasibleOptionFlags", &getFeasibleOptionFlags, "");

    m.def("getMidPriceQ", &getMidPriceQ, "");

    m.def("getMidPriceQReg", &getMidPriceQReg, "");

    m.def("getQReg", &getMidPriceQReg, "");
}
