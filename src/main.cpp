#include "optimizer.cpp"
#include "option_cleaner.cpp"
#include "option_cleaner.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

PYBIND11_MODULE(bindings, m) {
    m.doc() = ""; 

    // Expose functions
    // m.def("otm_payoff", &otm_payoff, "Compute out-of-the-money payoff");
    // m.def("omegaLMask", &omegaLMask, "Compute omegaL mask");
    m.def("performOptimization", &performOptimization, "Perform optimization");
    
    m.def("getFeasibleOptionFlags", &getFeasibleOptionFlags, "");

    m.def("getMidPriceQ", &getMidPriceQ, "");

    m.def("getMidPriceQReg", &getMidPriceQReg, "");

    m.def("getQReg", &getMidPriceQReg, "");
}
