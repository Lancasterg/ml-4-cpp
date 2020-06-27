#include <climits>

#ifndef SimpleLinearRegressor_H_
#define SimpleLinearRegressor_H_

#include <vector>
#include "../linear_algebra/linalg.h"
#include "AbstractRegressorModel.h"

namespace ml4cpp {

    template<class T>
    class SimpleLinearRegressor : public AbstractRegressorModel<T> {

    private:
        __unused  LinearAlgebra linalg;
        std::vector<double> coefficients;

        inline double updateWeights(std::vector<T> X, std::vector<T> Y, double learningRate);

    public:
        inline SimpleLinearRegressor(int i);

        inline void fit(std::vector<T> x, std::vector<T> y, double learningRate, int iterations);

        inline double meanSquaredError(std::vector<T> X, std::vector<T> Y);

        inline void setCoefficients(std::vector<T> coeff) override;

        inline double predict(const T &x) override;

    };
};


#endif
