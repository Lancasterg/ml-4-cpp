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
    template<class T>
    double AbstractBinaryLogisticRegressor<T>::sigmoid(T x) {
        return x / (1.0 + std::fabs(x));
    }

    /**
    * Make a probabilistic prediction
    * @param x: feature vector
    * @return The probability of being in class 1 in range 0-1
    */
    template<class T>
    double AbstractBinaryLogisticRegressor<T>::predictProba(std::vector<T> x) {
        return sigmoid(bias + ml4cpp::LinearAlgebra::dotProd(x, coefficients));
    }

    /**
    * Make a categorical prediction
    * @param x: Feature vector
    * @return The predicted class
    */
    template<class T>
    int AbstractBinaryLogisticRegressor<T>::predict(std::vector<T> x) {
        if (predictProba(x) > threshold) {
            return 1;
        } else {
            return 0;
        }
    }

    template<class T>
    void AbstractBinaryLogisticRegressor<T>::fit(std::vector<std::vector<T>> X, std::vector<T> Y) {
        // Do nothing
    }

    template<class T>
    void AbstractBinaryLogisticRegressor<T>::evaluate(std::vector<std::vector<T>> X, std::vector<T> Y) {
        // Do nothing
    }

    /**
     * Getter method for coefficients
     * @return
     */
    template<class T>
    std::vector<double> AbstractBinaryLogisticRegressor<T>::getCoefficients() {
        return coefficients;
    }


    /**
     * Constructor for BinaryLogisticRegressor
     * @param n_features: Number of features
     */
    template<class T>
    BinaryLogisticRegressorCm<T>::BinaryLogisticRegressorCm(int n_features) {
        this->bias = 1;
        this->coefficients = std::vector<double>(n_features);
        this->num_coefficients = n_features;
        this->threshold = 0.5;
    }


    /**
     * Fit the model
     * @param X: Fit the model using stochastic gradient descent
     * @param Y: Target classes
     */
    template<class T>
    void BinaryLogisticRegressorCm<T>::fit(std::vector<std::vector<T>> X, std::vector<T> Y) {

        double error, bias_deriv, coeff_deriv;
        double learningRate = 0.001;
        std::vector<double> x;
        int n_iters = 100;

        for (int iter = 0; iter < n_iters; iter++) {
            for (size_t i = 0; i < X[0].size(); i++) {
                x = ml4cpp::FileReader::getRow(i, X);

                error = Y[i] - this->predict(x);
                bias_deriv = -2 * this->bias * error;
                this->bias -= (bias_deriv) * learningRate;

                for (size_t j = 0; j < this->coefficients.size(); j++) {
                    coeff_deriv = -2 * x[j] * error;
                    this->coefficients[j] -= coeff_deriv * learningRate;
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
    template <class T>
    double BinaryLogisticRegressorCm<T>::crossEntropy(std::vector<std::vector<T>> X, std::vector<T> Y) {
        T cost_0 = 0;
        T cost_1 = 0;
        std::vector<T> x;

        for (size_t i = 0; i < X[0].size(); i++) {
            x = ml4cpp::FileReader::getRow(i, X);

            if (Y[i] == 1) {
                cost_1 += -log(this->predict(x));
            } else {
                cost_0 += -log(1 - this->predict(x));
            }
        }

        return (cost_0 + cost_1) / X[0].size();
    }


    /**
     * Evaluate the model
     * @param X: Feature vectors
     * @param Y: Target classes
     */
    template<class T>
    void BinaryLogisticRegressorCm<T>::evaluate(std::vector<std::vector<T>> X, std::vector<T> Y) {
        int n_correct = 0;
        std::vector<T> x;
        for (size_t i = 0; i < X[0].size(); i++) {
            x = ml4cpp::FileReader::getRow(i, X);
            if (this->predict(x) == Y[i]) {
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
    template<class T>
    void BinaryLogisticRegressorRm<T>::fit(std::vector<std::vector<T>> X, std::vector<T> Y) {

        double error, bias_deriv, coeff_deriv;
        double learningRate = 0.001;
        std::vector<double> x;
        int n_iters = 100;

        for (int iter = 0; iter < n_iters; iter++) {
            for (size_t i = 0; i < X.size(); i++) {
                x = X[i];

                error = Y[i] - this->predict(x);
                bias_deriv = -2 * this->bias * error;
                this->bias -= (bias_deriv) * learningRate;

                for (size_t j = 0; j < this->coefficients.size(); j++) {
                    coeff_deriv = -2 * x[j] * error;
                    this->coefficients[j] -= coeff_deriv * learningRate;
                }
            }
        }
    }

    /**
    * Evaluate the model
    * @param X: Feature vectors
    * @param Y: Target classes
    */
    template<class T>
    void BinaryLogisticRegressorRm<T>::evaluate(std::vector<std::vector<T>> X, std::vector<T> Y) {
        int n_correct = 0;
        for (size_t i = 0; i < X.size(); i++) {
            if (this->predict(X[i]) == Y[i]) {
                n_correct++;
            }
        }
        std::cout << "Accuracy: " << n_correct / (double) X.size() << std::endl;
    }

    /**
     * Constructor for row-major BinarBinaryLogisticRegressor
     * @param n_features
     */
    template<class T>
    BinaryLogisticRegressorRm<T>::BinaryLogisticRegressorRm(int n_features) {
        this->bias = 1;
        this->coefficients = std::vector<double>(n_features);
        this->num_coefficients = n_features;
        this->threshold = 0.5;
    }

    template<class T>
    double BinaryLogisticRegressorRm<T>::crossEntropy(std::vector<std::vector<T>> X, std::vector<T> Y) {
        return 0;
    }
}
