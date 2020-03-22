#ifndef SimpleLinearRegressor_H_
#define SimpleLinearRegressor_H_

#include <vector>
#include "../linear_algebra/linalg.h"
#include "AbstractRegressorModel.h"

namespace ml4cpp{

class SimpleLinearRegressor: public AbstractRegressorModel{

	private:
		std::vector<double> coefficients;
		LinearAlgebra linalg;

		double updateWeights(std::vector<double> X, std::vector<double> Y, double learningRate);

	public:
		SimpleLinearRegressor(){};
		SimpleLinearRegressor(int n_features);

		// Getter / setter methods
		void setCoefficients(std::vector<double> coeff){coefficients = coeff;};
		std::vector<double> getCoefficients(){return coefficients;};

		void fit(std::vector<double> x, std::vector<double> y, double learningRate, int iterations);

		double predict(double x);
		void gradientDescent(); // TODO??
		double meanSquaredError(std::vector<double> X, std::vector<double> Y);

	};
};


#endif
