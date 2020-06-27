#ifndef LINEAR_REGRESSION_ABSTRACTREGRESSORMODEL_H_
#define LINEAR_REGRESSION_ABSTRACTREGRESSORMODEL_H_

#include <vector>

namespace ml4cpp {

    template<class T>
    class AbstractRegressorModel {

    private:
        std::vector<double> coefficients;
    public:

        AbstractRegressorModel();

        virtual ~AbstractRegressorModel();

        virtual void setCoefficients(std::vector<T> coeff);

        std::vector<T> getCoefficients() { return coefficients; };

        // Make a prediction
        virtual double predict(const T &x);

        virtual // Fit the model
        void fit(std::vector<T> X, std::vector<T> Y);


    };

}

#endif /* LINEAR_REGRESSION_ABSTRACTREGRESSORMODEL_H_ */
