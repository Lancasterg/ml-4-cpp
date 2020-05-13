#include "BinaryLogisticRegressor.h"
#include "linalg.h"
#include <cstdlib>
#include <cmath>
#include <FileReader.h>

namespace ml4cpp {

    /**
    * Approximate the Sigmoid function
    * @param x: Input to Sigmoid function
    * @return Sigmoid approximation
    */
    double AbstractBinaryLogisticRegressor::sigmoid(double x) {
        return x / (1.0 + std::fabs(x));
    }

    /**
    * Make a probabilistic prediction
    * @param x: feature vector
    * @return The probability of being in class 1 in range 0-1
    */
    double AbstractBinaryLogisticRegressor::predictProba(std::vector<double> x) {
        return sigmoid(bias + ml4cpp::LinearAlgebra::dotProd(x, coefficients));
    }

    /**
    * Make a categorical prediction
    * @param x: Feature vector
    * @return The predicted class
    */
    int AbstractBinaryLogisticRegressor::predict(std::vector<double> x) {
        if (predictProba(x) > threshold) {
            return 1;
        } else {
            return 0;
        }
    }

    void AbstractBinaryLogisticRegressor::fit(std::vector<std::vector<double>> X, std::vector<double> Y) {
        // Do nothing
    }

    void AbstractBinaryLogisticRegressor::evaluate(std::vector<std::vector<double>> X, std::vector<double> Y) {
        // Do nothing
    }


    /**
     * Constructor for BinaryLogisticRegressor
     * @param n_features: Number of features
     */
    BinaryLogisticRegressorCm::BinaryLogisticRegressorCm(int n_features) {
        bias = 1;
        coefficients = std::vector<double>(n_features);
        num_coefficients = n_features;
        threshold = 0.5;
    }


    /**
     * Fit the model
     * @param X: Fit the model using stochastic gradient descent
     * @param Y: Target classes
     */
    void BinaryLogisticRegressorCm::fit(Matrix X, std::vector<double> Y) {

        double error, bias_deriv, coeff_deriv;
        double learningRate = 0.001;
        std::vector<double> x;
        int n_iters = 100;

        for (int iter = 0; iter < n_iters; iter++) {
            for (size_t i = 0; i < X[0].size(); i++) {
                x = ml4cpp::FileReader::getRow(i, X);

                error = Y[i] - predict(x);
                bias_deriv = -2 * bias * error;
                bias -= (bias_deriv) * learningRate;

                for (size_t j = 0; j < coefficients.size(); j++) {
                    coeff_deriv = -2 * x[j] * error;
                    coefficients[j] -= coeff_deriv * learningRate;
                }
            }
        }
    }




    /**
     * Calculate the cross-entropy
     * @param X: Feature vectors
     * @param Y: Target classes
     * @return The calculated cross-entropy
     */
    double BinaryLogisticRegressorCm::crossEntropy(Matrix X, std::vector<double> Y) {
        double cost_0 = 0;
        double cost_1 = 0;
        std::vector<double> x;

        for (size_t i = 0; i < X[0].size(); i++) {
            x = ml4cpp::FileReader::getRow(i, X);

            if (Y[i] == 1) {
                cost_1 += -log(predict(x));
            } else {
                cost_0 += -log(1 - predict(x));
            }
        }

        return (cost_0 + cost_1) / X[0].size();
    }


    /**
     * Evaluate the model
     * @param X: Feature vectors
     * @param Y: Target classes
     */
    void BinaryLogisticRegressorCm::evaluate(Matrix X, std::vector<double> Y) {
        int n_correct = 0;
        std::vector<double> x;
        for (size_t i = 0; i < X[0].size(); i++) {
            x = ml4cpp::FileReader::getRow(i, X);
            if (predict(x) == Y[i]) {
                n_correct++;
            }
        }

        std::cout << "Accuracy: " << n_correct / (double) X[0].size() << std::endl;
        std::cout << "Cross entropy: " << crossEntropy(X, Y) << std::endl;

    }


    /**
    * Fit on row-major data.
    * @param X: Feature vectors
    * @param Y: Target classes
    */
    void BinaryLogisticRegressorRm::fit(Matrix X, std::vector<double> Y) {

        double error, bias_deriv, coeff_deriv;
        double learningRate = 0.001;
        std::vector<double> x;
        int n_iters = 100;

        for (int iter = 0; iter < n_iters; iter++) {
            for (size_t i = 0; i < X.size(); i++) {
                x = X[i];

                error = Y[i] - predict(x);
                bias_deriv = -2 * bias * error;
                bias -= (bias_deriv) * learningRate;

                for (size_t j = 0; j < coefficients.size(); j++) {
                    coeff_deriv = -2 * x[j] * error;
                    coefficients[j] -= coeff_deriv * learningRate;
                }
            }
        }
    }

    /**
    * Evaluate the model
    * @param X: Feature vectors
    * @param Y: Target classes
    */
    void BinaryLogisticRegressorRm::evaluate(Matrix X, std::vector<double> Y) {
        int n_correct = 0;
        std::vector<double> x;
        for (size_t i = 0; i < X.size(); i++) {
            if (predict(X[i]) == Y[i]) {
                n_correct++;
            }
        }
        std::cout << "Accuracy: " << n_correct / (double) X.size() << std::endl;
    }

    /**
     * Constructor for row-major BinarBinaryLogisticRegressor
     * @param n_features
     */
    BinaryLogisticRegressorRm::BinaryLogisticRegressorRm(int n_features) {
        bias = 1;
        coefficients = std::vector<double>(n_features);
        num_coefficients = n_features;
        threshold = 0.5;
    }

    double BinaryLogisticRegressorRm::crossEntropy(std::vector<std::vector<double>> X, std::vector<double> Y) {
        return 0;
    }
}
