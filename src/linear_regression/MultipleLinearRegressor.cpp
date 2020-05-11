/*
 * MultipleLinearRegressor.cpp
 *
 *  Created on: 22 Mar 2020
 *      Author: George Lancaster
 */

#include "MultipleLinearRegressor.h"
#include <cmath>

namespace ml4cpp {


    /**
     * Initialise coefficients to 0, except for the first
     * @param n_features: number of features = number of coefficients
     */
    MultipleLinearRegressor::MultipleLinearRegressor(int n_features) {
        bias = 1;
        coefficients = std::vector<double>(n_features, 1);
    }

    /**
     * Set the coefficients for prediction
     * @param add_coeff: The additive coefficient
     * @param coeff: The multiplicative coefficients
     */
    void MultipleLinearRegressor::setCoefficients(double add_coeff, std::vector<double> coeff) {
        bias = add_coeff;
        coefficients = coeff;
    }

    /**
     * Make a prediction
     * @param X: Feature vector
     * @return result of prediction
     */
    double MultipleLinearRegressor::predict(const std::vector<double> &X) {
        return bias + ml4cpp::LinearAlgebra::dotProd(X, coefficients);
    }


    /**
     * Fit the model using stochastic gradient descent.
     * @param X: Training values
     * @param Y: Target values
     */
    void MultipleLinearRegressor::fit(Matrix X, std::vector<double> Y) {
        double error, bias_deriv, coeff_deriv;
        double learningRate = 0.001;
        std::vector<double> weightDerivatives(coefficients.size(), 0);
        int n_iters = 100;

        for (int iter = 0; iter < n_iters; iter++) {
            for (int i = 0; i < X[0].size(); i++) {
                std::vector<double> x = {X[0][i], X[1][i], X[2][i], X[3][i]};
                error = Y[i] - predict(x);
                bias_deriv = -2 * bias * error;
                bias -= (bias_deriv) * learningRate;
                for (int j = 0; j < weightDerivatives.size(); j++) {
                    coeff_deriv = -2 * x[j] * error;
                    coefficients[j] -= coeff_deriv * learningRate;
                }
            }
        }
    }

    double MultipleLinearRegressor::meanSquaredError(std::vector<std::vector<double>> X, std::vector<double> Y) {
        double error = 0;
        for (int i = 0; i < X.size(); i++) {
            std::vector<double> x = {X[i][0], X[i][1], X[i][2], X[i][3]};
            error += pow((Y[i] - predict(x)), 2);
        }
        return error / X.size();
    }
}

