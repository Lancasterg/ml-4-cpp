#ifndef SimpleLinearRegressor_H_
#define SimpleLinearRegressor_H_

#include <vector>
#include "../linear_algebra/linalg.h"
#include "AbstractRegressorModel.h"

namespace ml4cpp {

    class SimpleLinearRegressor : public AbstractRegressorModel {

    private:
        LinearAlgebra linalg;

        double updateWeights(std::vector<double> X, std::vector<double> Y, double learningRate);

    public:
        SimpleLinearRegressor() = default;

        explicit SimpleLinearRegressor(int n_features);


        void fit(std::vector<double> x, std::vector<double> y, double learningRate, int iterations);

        void gradientDescent(); // TODO??
        double meanSquaredError(std::vector<double> X, std::vector<double> Y);

        double predict(double x) override;

    };
};


#endif
