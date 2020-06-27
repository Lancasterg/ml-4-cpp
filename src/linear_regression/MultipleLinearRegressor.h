#ifndef LINEAR_REGRESSION_MULTIPLELINEARREGRESSOR_H_
#define LINEAR_REGRESSION_MULTIPLELINEARREGRESSOR_H_

#include <vector>
#include "../linear_algebra/linalg.h"
#include "AbstractRegressorModel.h"

namespace ml4cpp {

    template<class T>
    class MultipleLinearRegressor : public AbstractRegressorModel<T> {
    private:
        double bias;
        std::vector<double> coefficients;
        int num_coefficients;

    public:
        MultipleLinearRegressor() = default;

        explicit MultipleLinearRegressor(int n_features);

        double predict(const std::vector<T>& X);

        // Fit the model
        void fit(std::vector<std::vector<T>> X, std::vector<T> Y);

        void setCoefficients(double add_coeff, std::vector<double> coeff);

        double meanSquaredError(std::vector<std::vector<T>> X, std::vector<T> Y);
    };
}

#endif /* LINEAR_REGRESSION_MULTIPLELINEARREGRESSOR_H_ */
