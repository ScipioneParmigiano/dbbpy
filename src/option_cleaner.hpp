// /*
//  * Copyright 2021 <copyright holder> <email>
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#ifndef OPTION_CLEANER_HPP
#define OPTION_CLEANER_HPP

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

// Function to compute feasible option flags
Eigen::Matrix<bool, Eigen::Dynamic, 1> getFeasibleOptionFlags(
const Eigen::VectorXd& sp,
    const Eigen::VectorXd& bid,
    const Eigen::VectorXd& ask,
    const Eigen::VectorXd& strike,
    const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
    double spotsP, 
    double spbid,
    double spask
    );

// Function to compute mid price Q
Eigen::VectorXd getMidPriceQ(
    const Eigen::VectorXd& sp,
    const Eigen::VectorXd& bid,
    const Eigen::VectorXd& ask,
    const Eigen::VectorXd& strike,
    const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
    double spotsP, 
    double spbid,
    double spask    
    );

// Function to compute mid price Q with regularization
Eigen::VectorXd getMidPriceQReg(
    const Eigen::VectorXd& sp,
    const Eigen::VectorXd& bid,
    const Eigen::VectorXd& ask,
    const Eigen::VectorXd& strike,
    const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
    double spotsP, 
    double spbid,
    double spask);

// Function to compute Q with regularization
Eigen::VectorXd getQReg(
    const Eigen::VectorXd& bid,
    const Eigen::VectorXd& ask,
    const Eigen::VectorXd& strike,
    const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
    double spotsP, 
    double spbid,
    double spask
    );

#endif
