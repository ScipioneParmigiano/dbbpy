/*
 * Copyright 2021 <copyright holder> <email>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPTION_CLEANER_HPP
#define OPTION_CLEANER

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <Eigen/Dense>
#include "fusion.h"

typedef mosek::fusion::Matrix M_Matrix; 

typedef mosek::fusion::Variable M_Variable; 

typedef mosek::fusion::Var M_Var; 

typedef mosek::fusion::Expression M_Expression; 

typedef mosek::fusion::Domain M_Domain;

typedef monty::ndarray<double, 1> M_ndarray_1;

typedef monty::ndarray<double, 2> M_ndarray_2;

typedef mosek::fusion::Expr M_Expr; 

typedef mosek::fusion::Model::t M_Model; 

// template<typename T, typename... Args>
//     std::unique_ptr<T> make_unique(Args&&... args) {
//     return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
// }

// inline double  otm_payoff(double state, double strike, bool pFlag)  {
//     if(pFlag){
//         return(std::max(strike - state,0.0));
//     }
//     return(std::max(state - strike,0.0));
// }


// Function to compute feasible option flags
Eigen::Matrix<bool, Eigen::Dynamic, 1> getFeasibleOptionFlags(
    py::array_t<double> sp_py,
    py::array_t<double> bid_py,
    py::array_t<double> ask_py,
    py::array_t<double> strike_py,
    py::array_t<bool> pFlag_py,
    double spotsP, 
    double spbid,
    double spask
    );

// Function to compute mid price Q
Eigen::VectorXd getMidPriceQ(
    py::array_t<double> sp_py,
    py::array_t<double> bid_py,
    py::array_t<double> ask_py,
    py::array_t<double> strike_py,
    py::array_t<bool> pFlag_py,
    double spotsP, 
    double spbid,
    double spask
    );

// Function to compute mid price Q with regularization
Eigen::VectorXd getMidPriceQReg(
    py::array_t<double> sp_py,
    py::array_t<double> bid_py,
    py::array_t<double> ask_py,
    py::array_t<double> strike_py,
    py::array_t<bool> pFlag_py,
    double spotsP, 
    double spbid,
    double spask
    );

// Function to compute Q regression
Eigen::VectorXd getQReg(
    py::array_t<double> sp_py,
    py::array_t<double> bid_py,
    py::array_t<double> ask_py,
    py::array_t<double> strike_py,
    py::array_t<bool> pFlag_py,
    double spotsP, 
    double spbid,
    double spask
    );

#endif
