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

        virtual // Getter / setter methods
        void setCoefficients(std::vector<double> coeff) { coefficients = coeff; };

        std::vector<double> getCoefficients() { return coefficients; };

        // Make a prediction
        virtual double predict(double x);

        // Fit the model
        void fit(std::vector<double> X, std::vector<double> Y);


    };

}

#endif /* LINEAR_REGRESSION_ABSTRACTREGRESSORMODEL_H_ */
