/*
 * MultipleLinearRegressor.cpp
 *
 *  Created on: 22 Mar 2020
 *      Author: George Lancaster
 */

#include "MultipleLinearRegressor.h"

namespace ml4cpp {


    // - - - - - - - - - - - - - - - -
    // Initalise coefficients to 0, except first coeff
    // - - - - - - - - - - - - - - - -
    MultipleLinearRegressor::MultipleLinearRegressor(int n_features) {
        std::vector<double> coeffs(n_features, 0);
        coeffs[0] = 1;
        setCoefficients(coeffs);
    }

    // - - - - - - - - - - - - - - - -
    // Make a prediction
    // - - - - - - - - - - - - - - - -
    double MultipleLinearRegressor::predict(std::vector<double> X) {
        std::cout << X.size() << ", " << getNumCoefficients();

        return linalg.dotProd(X, getCoefficients());
    }


    // - - - - - - - - - - - - - - - -
    // Fit the model using stochastic gradient descent
    // - - - - - - - - - - - - - - - -
    void MultipleLinearRegressor::fit(Matrix X, std::vector<double> Y) {
        double error;
        std::vector<double> weightDerivatives(getNumCoefficients(), 0);
        int n_iters = 100;

        for (int i = 0; i < X[0].size(); i++) {

//			std::vector<double> abc;
//
//			for(int j = 0; j < X.size(); j++){
//
//				abc.push_back(X[i][j]);
//
//			}
//			error = Y[i] - predict(abc);
//			std::cout << error;

        }


    }


    double MultipleLinearRegressor::updateWeights(std::vector<double> X, std::vector<double> Y, double learningRate) {


        return 0;
    }

}

