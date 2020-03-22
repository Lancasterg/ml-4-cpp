#ifndef LINEAR_REGRESSION_ABSTRACTREGRESSORMODEL_H_
#define LINEAR_REGRESSION_ABSTRACTREGRESSORMODEL_H_

#include <vector>

namespace ml4cpp {

	class AbstractRegressorModel {
	private:

		std::vector<double> coefficients;


	public:


		AbstractRegressorModel();
		virtual ~AbstractRegressorModel();

		// Getter / setter methods
		void setCoefficients(std::vector<double> coeff){coefficients = coeff;};
		std::vector<double> getCoefficients(){return coefficients;};

		// Make a prediction
//		double predict(double x) = 0;



	};

}

#endif /* LINEAR_REGRESSION_ABSTRACTREGRESSORMODEL_H_ */
