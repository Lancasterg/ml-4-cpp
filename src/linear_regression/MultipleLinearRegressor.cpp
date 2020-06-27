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
    template<class T>
    MultipleLinearRegressor<T>::MultipleLinearRegressor(int n_features) {
        bias = 1;
        coefficients = std::vector<T>(n_features, 1);
        num_coefficients = n_features;
    }

    /**
     * Set the coefficients for prediction
     * @param add_coeff: The additive coefficient
     * @param coeff: The multiplicative coefficients
     */
    template<class T>
    void MultipleLinearRegressor<T>::setCoefficients(double add_coeff, std::vector<double> coeff) {
        bias = add_coeff;
        coefficients = std::move(coeff);
        num_coefficients = int(coefficients.size());
    }

    /**
     * Make a prediction
     * @param X: Feature vector
     * @return result of prediction
     */
    template<class T>
    double MultipleLinearRegressor<T>::predict(const std::vector<T> &X) {
        return bias + ml4cpp::LinearAlgebra::dotProd(X, coefficients);
    }


    /**
     * Fit the model using stochastic gradient descent.
     * @param X: Training values
     * @param Y: Target values
     */
    template<class T>
    void MultipleLinearRegressor<T>::fit(std::vector<std::vector<T>> X, std::vector<T> Y) {
        double error, bias_deriv, coeff_deriv;
        double learningRate = 0.001;
        std::vector<T> x;
        int n_iters = 100;

        for (int iter = 0; iter < n_iters; iter++) {
            for (size_t i = 0; i < X.size(); i++) {

                error = Y[i] - predict(X[i]);
                bias_deriv = -2 * bias * error;
                bias -= (bias_deriv) * learningRate;

                for (size_t j = 0; j < coefficients.size(); j++) {
                    coeff_deriv = -2 * X[i][j] * error;
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
    template<class T>
    double MultipleLinearRegressor<T>::meanSquaredError(std::vector<std::vector<T>> X, std::vector<T> Y) {
        double error = 0;
        for (size_t i = 0; i < X.size(); i++) {
            error += pow(Y[i] - predict(X[i]), 2);
        }
        return error / X.size();
    }
}

