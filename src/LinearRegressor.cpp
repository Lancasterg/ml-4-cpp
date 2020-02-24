/*
 * LinearRegressor.cpp
 *
 *  Created on: 21 Feb 2020
 *      Author: george
 */

#include "LinearRegressor.h"
#include <math.h>
#include <stdlib.h>


LinearRegressor::LinearRegressor(int n_features) {
	vector<double> coeffs(n_features + 1, 0);
	coeffs[0] = 5000;
	coeffs[1] = 0;
	setCoefficients(coeffs);
}

double LinearRegressor::predict(double x){
	return coefficients[0] + coefficients[1] * x;
}

double LinearRegressor::meanSquaredError(vector<double> X, vector<double> Y){
	double error = 0;
	for (int i = 0; i < X.size(); i++){
		error += pow((Y[i] - predict(X[i])), 2);
	}
	return error / X.size();
}

double LinearRegressor::updateWeights(vector<double> X, vector<double> Y, double learningRate){
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

	coefficients[0] -= (bias_deriv / X.size()) * learningRate;
	coefficients[1] -= (weight_deriv / X.size()) * learningRate;
	mse /= X.size();

	return mse;
}


void LinearRegressor::fit(vector<double> X, vector<double> Y, double learningRate, int iterations){
	double mse = 0;
	double prev_mse = -1;

	for (int i = 0; i < iterations; i++){
		mse = updateWeights(X, Y, learningRate);

//		printf("%f:%f\n", mse, prev_mse);
		if (mse == prev_mse){
			cout << "woop";
			return;
		}
		prev_mse = mse;
	}


}


