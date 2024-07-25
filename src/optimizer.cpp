// #include "optimizer.hpp"
// #include "utils.hpp"

// namespace py = pybind11;

// // Function to perform optimization
// py::tuple performOptimization(int n, double alpha, double lambda,
//                          py::array_t<int> omega_l_py,
//                          py::array_t<double> sp_py,
//                          py::array_t<double> strike_py,
//                          py::array_t<double> bid_py,
//                          py::array_t<double> ask_py,
//                          py::array_t<int> pFlag_py
//                          ) {

//     // Convert all the PyBind11 arrays to Eigen types
//     Eigen::Matrix<int, Eigen::Dynamic, 1> omega_l = numpy_vec_to_eigen<int>(omega_l_py);
//     Eigen::Matrix<double, Eigen::Dynamic, 1> sp_eigen = numpy_vec_to_eigen<double>(sp_py);
//     Eigen::Matrix<double, Eigen::Dynamic, 1> strike_eigen = numpy_vec_to_eigen<double>(strike_py);
//     Eigen::Matrix<double, Eigen::Dynamic, 1> bid_eigen = numpy_vec_to_eigen<double>(bid_py);
//     Eigen::Matrix<double, Eigen::Dynamic, 1> ask_eigen = numpy_vec_to_eigen<double>(ask_py);
//     Eigen::Matrix<int, Eigen::Dynamic, 1> pFlag_eigen = numpy_vec_to_eigen<int>(pFlag_py);

//     // Initialize payoff matrix and compute payoffs
//     size_t spLen = sp_eigen.size();
//     size_t optLen = bid_eigen.size();
//     Eigen::MatrixXd payoff_matrix(optLen, spLen);

//     // Fill the payoff matrix
//     for (size_t i = 0; i < optLen; ++i) {
//         for (size_t j = 0; j < spLen; ++j) {
//             // Compute OTM payoff based on spot, strike, and option type
//             payoff_matrix(i, j) = otm_payoff(sp_eigen(j), strike_eigen(i), pFlag_eigen(i)) / (0.5 * (bid_eigen[i] + ask_eigen[i]));
//         }
//     }

//     // Initialize the MOSEK Fusion model
//     mosek::fusion::Model::t M = new mosek::fusion::Model("main");
//     auto _M = monty::finally([&]() { M->dispose(); });

//     // Define variables P and Q
//     mosek::fusion::Variable::t p = M->variable("P", n, mosek::fusion::Domain::greaterThan(0.0));  // mosek::fusion::Variable P
//     mosek::fusion::Variable::t q = M->variable("Q", n, mosek::fusion::Domain::greaterThan(0.0));  // mosek::fusion::Variable Q

//     // Add constraints (make the P and Q congruent distributions)
//     M->constraint(mosek::fusion::Expr::sum(p), mosek::fusion::Domain::equalsTo(1.0));  // Sum of p elements equals 1
//     M->constraint(mosek::fusion::Expr::sum(q), mosek::fusion::Domain::equalsTo(1.0));  // Sum of q elements equals 1

//     // Add constraints involving payoff_matrix and q
//     Eigen::VectorXd result_bid = bid_eigen.array() / (0.5 * (bid_eigen.array() + ask_eigen.array()));
//     Eigen::VectorXd result_ask = ask_eigen.array() / (0.5 * (bid_eigen.array() + ask_eigen.array()));
//     mosek::fusion::Matrix::t payoff_monty_matr = mosek::fusion::Matrix::dense(eigenToStdMatrix<double>(payoff_matrix));
    
//     mosek::fusion::Expression::t product = mosek::fusion::Expr::mul(q, payoff_monty_matr);
//     M->constraint("bid_", product, mosek::fusion::Domain::greaterThan(eigenToStdVector<double>(result_bid)));
//     M->constraint("ask_", product, mosek::fusion::Domain::lessThan(eigenToStdVector<double>(result_ask)));

//     // Constraints for second moment of pricing kernel
//     mosek::fusion::Variable::t q_square = M->variable(n, mosek::fusion::Domain::greaterThan(0.0));
//     mosek::fusion::Variable::t p_square = M->variable(n, mosek::fusion::Domain::greaterThan(0.0));
//     mosek::fusion::Variable::t one = M->variable(1, mosek::fusion::Domain::equalsTo(1.0));

//     Eigen::VectorXd ones_vector = Eigen::VectorXd::Ones(n);
//     auto ones_ptr = std::make_shared<monty::ndarray<double, 1>>(ones_vector.data(), ones_vector.size()); 
//     mosek::fusion::Variable::t ones = M->variable(n, mosek::fusion::Domain::equalsTo(ones_ptr));
//     M->constraint("q_square", mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(q_square, 0.5), ones, q), mosek::fusion::Domain::inRotatedQCone());
//     M->constraint("p_square", mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(p_square, 0.5), ones, p), mosek::fusion::Domain::inRotatedQCone());

//     mosek::fusion::Variable::t u = M->variable(n);
//     M->constraint(mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(u, 0.5), p, q), mosek::fusion::Domain::inRotatedQCone());
//     M->constraint(mosek::fusion::Expr::sum(u), mosek::fusion::Domain::lessThan(alpha));

//     // Variance constraint using dot product
//     std::vector<double> gross_returns = computeGrossReturns(payoff_matrix);
//     std::shared_ptr<monty::ndarray<double, 1>> payoff (new monty::ndarray<double, 1>(n));
//     for (int i = 0; i < n; ++i) {
//         (*payoff)[i] = gross_returns[i] - log(gross_returns[i]) - 1;
//     }

//     mosek::fusion::Expression::t p_var = mosek::fusion::Expr::dot(payoff, p);
//     mosek::fusion::Expression::t q_var = mosek::fusion::Expr::dot(payoff, q);

//     M->constraint(mosek::fusion::Expr::sub(p_var, q_var), mosek::fusion::Domain::lessThan(0.0));

//     mosek::fusion::Variable::t p_vari = M->variable(1, mosek::fusion::Domain::greaterThan(0.0));
//     mosek::fusion::Variable::t q_vari = M->variable(1, mosek::fusion::Domain::greaterThan(0.0));
//     M->constraint(mosek::fusion::Expr::sub(p_var, p_vari), mosek::fusion::Domain::equalsTo(0.0));
//     M->constraint(mosek::fusion::Expr::sub(q_var, q_vari), mosek::fusion::Domain::equalsTo(0.0));

//     // Define objective function using mask omega_l
//     std::shared_ptr<monty::ndarray<double, 1>> mask = omegaLMask(omega_l, n);
    
//     mosek::fusion::Expression::t obj_expr = mosek::fusion::Expr::add(mosek::fusion::Expr::dot(mask, p), mosek::fusion::Expr::dot(mask, q));
//     mosek::fusion::Expression::t regularization = mosek::fusion::Expr::add(mosek::fusion::Expr::sum(p_square), mosek::fusion::Expr::sum(q_square));
//     mosek::fusion::Expression::t obj_expr_reg = mosek::fusion::Expr::sub(obj_expr, mosek::fusion::Expr::mul(lambda, regularization));

//     M->objective("obj", mosek::fusion::ObjectiveSense::Maximize, obj_expr_reg);

//     // Solve the problem
//     M->solve();

//     // Print solution
//     // auto sol_p = p->level();
//     // auto sol_q = q->level();
//     // std::cout << "Solution:" << std::endl;
//     // std::cout << "P = " << *sol_p << std::endl;
//     // std::cout << "Q = " << *sol_q << std::endl;

//     auto p_ptr = p->level();
//     auto q_ptr = q->level();

//     std::vector<double> p_vec(p_ptr->size());
//     std::vector<double> q_vec(q_ptr->size());

//     std::copy(p_ptr->begin(), p_ptr->end(), p_vec.begin());
//     std::copy(q_ptr->begin(), q_ptr->end(), q_vec.begin());

//     // Create NumPy arrays from the vectors
//     py::array_t<double> p_np = py::array_t<double>(p_vec.size(), p_vec.data());
//     py::array_t<double> q_np = py::array_t<double>(q_vec.size(), q_vec.data());

//     return py::make_tuple(p_np, q_np);
// }



#include "optimizer.hpp"
#include "utils.hpp"

namespace py = pybind11;

// Function to perform optimization
py::tuple performOptimization(int n, double alpha, double lambda,
                              const Eigen::MatrixXi& omega_l_eigen,
                              const Eigen::MatrixXd& sp_eigen,
                              const Eigen::MatrixXd& strike_eigen,
                              const Eigen::MatrixXd& bid_eigen,
                              const Eigen::MatrixXd& ask_eigen,
                              const Eigen::MatrixXi& pFlag_eigen) {

    // Initialize payoff matrix and compute payoffs
    size_t spLen = sp_eigen.cols();
    size_t optLen = bid_eigen.rows();
    Eigen::MatrixXd payoff_matrix(optLen, spLen);

    // Fill the payoff matrix
    for (size_t i = 0; i < optLen; ++i) {
        for (size_t j = 0; j < spLen; ++j) {
            // Compute OTM payoff based on spot, strike, and option type
            payoff_matrix(i, j) = otm_payoff(sp_eigen(i, j), strike_eigen(i, 0), pFlag_eigen(i, 0)) / (0.5 * (bid_eigen(i, 0) + ask_eigen(i, 0)));
        }
    }

    // Initialize the MOSEK Fusion model
    mosek::fusion::Model::t M = new mosek::fusion::Model("main");
    auto _M = monty::finally([&]() { M->dispose(); });

    // Define variables P and Q
    mosek::fusion::Variable::t p = M->variable("P", n, mosek::fusion::Domain::greaterThan(0.0));  // mosek::fusion::Variable P
    mosek::fusion::Variable::t q = M->variable("Q", n, mosek::fusion::Domain::greaterThan(0.0));  // mosek::fusion::Variable Q

    // Add constraints (make the P and Q congruent distributions)
    M->constraint(mosek::fusion::Expr::sum(p), mosek::fusion::Domain::equalsTo(1.0));  // Sum of p elements equals 1
    M->constraint(mosek::fusion::Expr::sum(q), mosek::fusion::Domain::equalsTo(1.0));  // Sum of q elements equals 1

    // Add constraints involving payoff_matrix and q
    Eigen::VectorXd result_bid = bid_eigen.col(0).array() / (0.5 * (bid_eigen.col(0).array() + ask_eigen.col(0).array()));
    Eigen::VectorXd result_ask = ask_eigen.col(0).array() / (0.5 * (bid_eigen.col(0).array() + ask_eigen.col(0).array()));
    mosek::fusion::Matrix::t payoff_monty_matr = mosek::fusion::Matrix::dense(eigenToStdMatrix<double>(payoff_matrix));

    mosek::fusion::Expression::t product = mosek::fusion::Expr::mul(q, payoff_monty_matr);
    M->constraint("bid_", product, mosek::fusion::Domain::greaterThan(eigenToStdVector<double>(result_bid)));
    M->constraint("ask_", product, mosek::fusion::Domain::lessThan(eigenToStdVector<double>(result_ask)));

    // Constraints for second moment of pricing kernel
    mosek::fusion::Variable::t q_square = M->variable(n, mosek::fusion::Domain::greaterThan(0.0));
    mosek::fusion::Variable::t p_square = M->variable(n, mosek::fusion::Domain::greaterThan(0.0));
    mosek::fusion::Variable::t one = M->variable(1, mosek::fusion::Domain::equalsTo(1.0));

    Eigen::VectorXd ones_vector = Eigen::VectorXd::Ones(n);
    auto ones_ptr = std::make_shared<monty::ndarray<double, 1>>(ones_vector.data(), ones_vector.size());
    mosek::fusion::Variable::t ones = M->variable(n, mosek::fusion::Domain::equalsTo(ones_ptr));
    M->constraint("q_square", mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(q_square, 0.5), ones, q), mosek::fusion::Domain::inRotatedQCone());
    M->constraint("p_square", mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(p_square, 0.5), ones, p), mosek::fusion::Domain::inRotatedQCone());

    mosek::fusion::Variable::t u = M->variable(n);
    M->constraint(mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(u, 0.5), p, q), mosek::fusion::Domain::inRotatedQCone());
    M->constraint(mosek::fusion::Expr::sum(u), mosek::fusion::Domain::lessThan(alpha));

    // Variance constraint using dot product
    std::vector<double> gross_returns = computeGrossReturns(payoff_matrix);
    std::shared_ptr<monty::ndarray<double, 1>> payoff(new monty::ndarray<double, 1>(n));
    for (int i = 0; i < n; ++i) {
        (*payoff)[i] = gross_returns[i] - log(gross_returns[i]) - 1;
    }

    mosek::fusion::Expression::t p_var = mosek::fusion::Expr::dot(payoff, p);
    mosek::fusion::Expression::t q_var = mosek::fusion::Expr::dot(payoff, q);

    M->constraint(mosek::fusion::Expr::sub(p_var, q_var), mosek::fusion::Domain::lessThan(0.0));

    mosek::fusion::Variable::t p_vari = M->variable(1, mosek::fusion::Domain::greaterThan(0.0));
    mosek::fusion::Variable::t q_vari = M->variable(1, mosek::fusion::Domain::greaterThan(0.0));
    M->constraint(mosek::fusion::Expr::sub(p_var, p_vari), mosek::fusion::Domain::equalsTo(0.0));
    M->constraint(mosek::fusion::Expr::sub(q_var, q_vari), mosek::fusion::Domain::equalsTo(0.0));

    // Define objective function using mask omega_l
    std::shared_ptr<monty::ndarray<double, 1>> mask = omegaLMask(omega_l_eigen, n);

    mosek::fusion::Expression::t obj_expr = mosek::fusion::Expr::add(mosek::fusion::Expr::dot(mask, p), mosek::fusion::Expr::dot(mask, q));
    mosek::fusion::Expression::t regularization = mosek::fusion::Expr::add(mosek::fusion::Expr::sum(p_square), mosek::fusion::Expr::sum(q_square));
    mosek::fusion::Expression::t obj_expr_reg = mosek::fusion::Expr::sub(obj_expr, mosek::fusion::Expr::mul(lambda, regularization));

    M->objective("obj", mosek::fusion::ObjectiveSense::Maximize, obj_expr_reg);

    // Solve the problem
    M->solve();

    // Retrieve and convert solution
    auto p_ptr = p->level();
    auto q_ptr = q->level();

    std::vector<double> p_vec(p_ptr->size());
    std::vector<double> q_vec(q_ptr->size());

    std::copy(p_ptr->begin(), p_ptr->end(), p_vec.begin());
    std::copy(q_ptr->begin(), q_ptr->end(), q_vec.begin());

    // Create NumPy arrays from the vectors
    py::array_t<double> p_np(p_vec.size(), p_vec.data());
    py::array_t<double> q_np(q_vec.size(), q_vec.data());

    return py::make_tuple(p_np, q_np);
}
