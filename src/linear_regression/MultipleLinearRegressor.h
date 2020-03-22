#ifndef LINEAR_REGRESSION_MULTIPLELINEARREGRESSOR_H_
#define LINEAR_REGRESSION_MULTIPLELINEARREGRESSOR_H_

#include <vector>
#include "../linear_algebra/linalg.h"
#include "AbstractRegressorModel.h"

namespace ml4cpp{

	class MultipleLinearRegressor: public AbstractRegressorModel{
	private:
		std::vector<double> coefficients;
		LinearAlgebra linalg;

		double updateWeights(std::vector<double> X, std::vector<double> Y, double learningRate);
	public:
		MultipleLinearRegressor(){};
		MultipleLinearRegressor(int n_features);

		int getNumCoefficients(){ return getCoefficients().size(); }

		// Make a predicition
		double predict(std::vector<double> X);


		// Fit the model
		void fit(Matrix X, std::vector<double> Y);

	};
}

#endif /* LINEAR_REGRESSION_MULTIPLELINEARREGRESSOR_H_ */
