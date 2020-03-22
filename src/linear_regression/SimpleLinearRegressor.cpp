/*
 * SimpleLinearRegressor.cpp
 *
 *  Created on: 21 Feb 2020
 *      Author: George Lancaster
 */

#include "SimpleLinearRegressor.h"
#include <math.h>
#include <stdlib.h>

namespace ml4cpp{

	SimpleLinearRegressor::SimpleLinearRegressor(int n_features) {
		std::vector<double> coeffs(n_features + 1, 0);
		coeffs[0] = 5000;
		coeffs[1] = 0;
		setCoefficients(coeffs);
	}

	// - - - - - - - - - - - - - - - -
	// Make a prediction
	// - - - - - - - - - - - - - - - -
	double SimpleLinearRegressor::predict(double x){

		return getCoefficients()[0] + getCoefficients()[1] * x;
	}

	// - - - - - - - - - - - - - - - -
	// Calculate MSE over entire data set
	// - - - - - - - - - - - - - - - -
	double SimpleLinearRegressor::meanSquaredError(std::vector<double> X, std::vector<double> Y){
		double error = 0;
		for (int i = 0; i < X.size(); i++){
			error += pow((Y[i] - predict(X[i])), 2);
		}
		return error / X.size();
	}


	// - - - - - - - - - - - - - - - -
	// Update weights using gradient descent
	// - - - - - - - - - - - - - - - -
	double SimpleLinearRegressor::updateWeights(std::vector<double> X, std::vector<double> Y, double learningRate){
		double bias_deriv = 0;
		double weight_deriv = 0;
		double error = 0;
		double mse = 0;

		for (int i = 0; i < X.size(); i++){
			error = Y[i] - predict(X[i]);
			bias_deriv += -2 * error;
			weight_deriv += -2 * X[i] * error;
			mse += error;
		}

		getCoefficients()[0] -= (bias_deriv / X.size()) * learningRate;
		getCoefficients()[1] -= (weight_deriv / X.size()) * learningRate;
		mse /= X.size();

		return mse;
	}

	// - - - - - - - - - - - - - - - -
	// Fit the linear regression model
	// - - - - - - - - - - - - - - - -
	void SimpleLinearRegressor::fit(std::vector<double> X, std::vector<double> Y, double learningRate, int iterations){
		double mse = 0;
		double prev_mse = -1;

		for (int i = 0; i < iterations; i++){
			mse = updateWeights(X, Y, learningRate);

			if (mse == prev_mse){
				return;
			}
			prev_mse = mse;
		}


	}

};


