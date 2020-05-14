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

        virtual void setCoefficients(std::vector<double> coeff);

        std::vector<double> getCoefficients() { return coefficients; };

        // Make a prediction
        virtual double predict(const double &x);

        // Fit the model
        void fit(std::vector<double> X, std::vector<double> Y);


    };

}

#endif /* LINEAR_REGRESSION_ABSTRACTREGRESSORMODEL_H_ */
