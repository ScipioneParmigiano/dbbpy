#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

// Function to compute out-of-the-money (OTM) payoff
double otm_payoff(double spot, double strike, bool isPut) {
    if (isPut) {
        return std::max(strike - spot, 0.0);
    } else {
        return std::max(spot - strike, 0.0);
    }
}

// Function to compute omega L Mask
std::shared_ptr<monty::ndarray<double, 1>> omegaLMask(const Eigen::VectorXd& positions, int n) {
    // Initialize an n-long array with all elements set to 0.0
    std::vector<double> x(n);
    auto result = monty::new_array_ptr<double>(x);    
    std::fill(result->begin(), result->end(), 0.0);

    // Set specified positions to 1.0
    for (int i = 0; i < positions.size(); ++i) {
        int pos = positions(i);
        if (pos < n && pos >= 0) {  // Ensure the position is within bounds
            (*result)(pos) = 1.0;
        }
    }

    return result;
}

template<typename T>
// Create a shared_ptr for vectors of the appropriate type
std::shared_ptr<monty::ndarray<T, 1>> eigenToStdVector(const Eigen::VectorXd& eigenVec) {
    auto stdVec = std::make_shared<monty::ndarray<T, 1>>(eigenVec.data(), eigenVec.size());
    
    return stdVec;
}

template<typename T>
// Create a shared_ptr for matrices of the appropriate type
std::shared_ptr<monty::ndarray<T, 2>> eigenToStdMatrix(const Eigen::MatrixXd& eigenMat) {
    auto stdMat = std::make_shared<monty::ndarray<T, 2>>(eigenMat.data(), monty::shape(eigenMat.cols(), eigenMat.rows()));
    return stdMat;
}

// Function to compute gross returns from a payoff matrix
std::vector<double> computeGrossReturns(const Eigen::MatrixXd& payoff_matrix) {
    // Compute gross returns using Eigen's vectorized operations
    Eigen::VectorXd gross_returns = payoff_matrix.rowwise().sum();

    // Convert Eigen::VectorXd to std::vector<double>
    return std::vector<double>(gross_returns.data(), gross_returns.data() + gross_returns.size());
}

#endif
