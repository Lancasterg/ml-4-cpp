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

		// Getter / setter methods
		void setCoefficients(std::vector<double> coeff){coefficients = coeff;};
		std::vector<double> getCoefficients(){return coefficients;};

	};
}

#endif /* LINEAR_REGRESSION_MULTIPLELINEARREGRESSOR_H_ */
