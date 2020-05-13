/*
 * MultipleLinearRegressor.cpp
 *
 *  Created on: 22 Mar 2020
 *      Author: George Lancaster
 */

#include "MultipleLinearRegressor.h"
#include "FileReader.h"
#include <cmath>
#include <utility>

namespace ml4cpp {


    /**
     * Initialise coefficients to 0, except for the first
     * @param n_features: number of features = number of coefficients
     */
    MultipleLinearRegressor::MultipleLinearRegressor(int n_features) {
        bias = 1;
        coefficients = std::vector<double>(n_features, 1);
        num_coefficients = n_features;
    }

    /**
     * Set the coefficients for prediction
     * @param add_coeff: The additive coefficient
     * @param coeff: The multiplicative coefficients
     */
    void MultipleLinearRegressor::setCoefficients(double add_coeff, std::vector<double> coeff) {
        bias = add_coeff;
        coefficients = std::move(coeff);
        num_coefficients = int(coefficients.size());
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
        std::vector<double> x;
        int n_iters = 100;

        for (int iter = 0; iter < n_iters; iter++) {
            for (size_t i = 0; i < X[0].size(); i++) {
                x = ml4cpp::FileReader::getRow(i, X);
                error = Y[i] - predict(x);
                bias_deriv = -2 * bias * error;
                bias -= (bias_deriv) * learningRate;

                for (size_t j = 0; j < coefficients.size(); j++) {
                    coeff_deriv = -2 * x[j] * error;
                    coefficients[j] -= coeff_deriv * learningRate;
                }
            }
        }
    }

    /**
     * Calculate the mean squared error over a dataset.
     * @param X: Feature vector
     * @param Y: Target predictions
     * @return
     */
    double MultipleLinearRegressor::meanSquaredError(std::vector<std::vector<double>> X, std::vector<double> Y) {
        double error = 0;
        std::vector<double> x;

        for (size_t i = 0; i < X[0].size(); i++) {
            x = {X[0][i], X[1][i], X[2][i], X[3][i]};
            error += pow(Y[i] - predict(x), 2);
        }
        return error / X[0].size();
    }
}

