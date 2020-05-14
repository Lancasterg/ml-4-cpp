#ifndef SimpleLinearRegressor_H_
#define SimpleLinearRegressor_H_

#include <vector>
#include "../linear_algebra/linalg.h"
#include "AbstractRegressorModel.h"

namespace ml4cpp {

    class SimpleLinearRegressor : public AbstractRegressorModel {

    private:
        LinearAlgebra linalg;
        std::vector<double> coefficients;

        double updateWeights(std::vector<double> X, std::vector<double> Y, double learningRate);

    public:
        SimpleLinearRegressor();

        explicit SimpleLinearRegressor(int n_features);

        void fit(std::vector<double> x, std::vector<double> y, double learningRate, int iterations);

        double meanSquaredError(std::vector<double> X, std::vector<double> Y);

        void setCoefficients(std::vector<double> coeff) override;

        double predict(const double &x) override;

    };
};


#endif
