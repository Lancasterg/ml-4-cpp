/*
 * SimpleLinearRegressor.cpp
 *
 *  Created on: 21 Feb 2020
 *      Author: George Lancaster
 */

#include "SimpleLinearRegressor.h"
#include <cmath>

namespace ml4cpp {

    template<class T>
    SimpleLinearRegressor<T>::SimpleLinearRegressor(int i) {
        coefficients = {1, 1};
    }

    /**
     * Make a prediction
     * @param x: Input value
     * @return:: Output prediction
     */
    template<class T>
    double SimpleLinearRegressor<T>::predict(const T &x) {
        return coefficients[0] + coefficients[1] * x;
    }


    /**
     * Calculate MSE over entire data set
     * @param X: Training values
     * @param Y: Target values
     * @return
     */
     template<class T>
    double SimpleLinearRegressor<T>::meanSquaredError(std::vector<T> X, std::vector<T> Y) {
        double error = 0;
        for (size_t i = 0; i < X.size(); i++) {
            error += pow((Y[i] - predict(X[i])), 2);
        }
        return error / X.size();
    }


    /**
     * Update weights using gradient descent
     * @param X: Training data
     * @param Y: Target values
     * @param learningRate: Rate at which to update weights
     * @return
     */
     template<class T>
    double SimpleLinearRegressor<T>::updateWeights(std::vector<T> X, std::vector<T> Y, double learningRate) {
        double bias_deriv = 0;
        double weight_deriv = 0;
        double error = 0;
        double mse = 0;

        for (size_t i = 0; i < X.size(); i++) {
            error = Y[i] - predict(X[i]);
            bias_deriv += -2 * error;
            weight_deriv += -2 * X[i] * error;
            mse += error;
        }

        coefficients[0] -= (bias_deriv / X.size()) * learningRate;
        coefficients[1] -= (weight_deriv / X.size()) * learningRate;
        mse /= X.size();

        return mse;
    }

    /**
     * Fit the linear regression model
     * @param X: Training data
     * @param Y: Target values
     * @param learningRate: Rate at which to update weights
     * @param iterations: Number of training iterations
     */
    template <class T>
    void SimpleLinearRegressor<T>::fit(std::vector<T> X, std::vector<T> Y, double learningRate, int iterations) {
        double mse = 0;
        double prev_mse = -1;

        for (int i = 0; i < iterations; i++) {
            mse = updateWeights(X, Y, learningRate);

            if (mse == prev_mse) {
                return;
            }
            prev_mse = mse;
        }
    }

    template<class T>
    void SimpleLinearRegressor<T>::setCoefficients(std::vector<T> coeff) {
        coefficients = coeff;
    }

};


