#ifndef LINEAR_REGRESSION_MULTIPLELINEARREGRESSOR_H_
#define LINEAR_REGRESSION_MULTIPLELINEARREGRESSOR_H_

#include <vector>
#include "../linear_algebra/linalg.h"
#include "AbstractRegressorModel.h"

namespace ml4cpp {

    class MultipleLinearRegressor : public AbstractRegressorModel {
    private:
        double bias;
        std::vector<double> coefficients;

        double updateWeights(std::vector<double> X, std::vector<double> Y, double learningRate);

    public:
        MultipleLinearRegressor() = default;

        explicit MultipleLinearRegressor(int n_features);

        // Make a predicition
        double predict(const std::vector<double>& X);

        // Fit the model
        void fit(Matrix X, std::vector<double> Y);

        void setCoefficients(double add_coeff, std::vector<double> coeff);

        double meanSquaredError(Matrix X, std::vector<double> Y);
    };
}

#endif /* LINEAR_REGRESSION_MULTIPLELINEARREGRESSOR_H_ */
