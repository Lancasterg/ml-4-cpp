#ifndef LINEARREGRESSOR_H_
#define LINEARREGRESSOR_H_

#include<vector>
#include"linalg.h"

using namespace std;

class LinearRegressor
{
private:
	vector<double> coefficients;
	LinearAlgebra linalg;

	double updateWeights(vector<double> X, vector<double> Y, double learningRate);

public:
	LinearRegressor(){};
	LinearRegressor(int n_features);

	// Getter / setter methods
	void setCoefficients(vector<double> coeff){coefficients = coeff;};
	vector<double> getCoefficients(){return coefficients;};

	void fit(vector<double> x, vector<double> y, double learningRate, int iterations);

	double predict(double x);
	void gradientDescent();
	double meanSquaredError(vector<double> X, vector<double> Y);



};



#endif
