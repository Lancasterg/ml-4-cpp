#include <FileReader.h>
#include "../libs/catch.hpp"
#include "../logistic_regression/BinaryLogisticRegressor.h"
#include "../logistic_regression/BinaryLogisticRegressor.cpp"

/**
 * Return some simple data to test fitting and evaluation
 * in row-major format.
 * @return Vector of feature vectors with class = 1 or class = 0
 */
Matrix getTestDataRm() {
    return {
            // Class 1
            {0.95,  1.01,  1.10,  1.00},
            {0.94,  1.01,  1.11,  1.00},
            {0.96,  0.99,  1.12,  1.00},
            {0.95,  1.01,  1.11,  1.00},
            {0.96,  1.02,  1.10,  1.00},

            // Class 0
            {100.0, 200.0, 300.0, 0},
            {100.1, 200.1, 300.1, 0},
            {100.2, 200.1, 300.0, 0},
            {99.99, 199.9, 299.9, 0},
            {99.98, 199.9, 299.9, 0}
    };
}

/**
 *  Return some simple data to test fitting and evaluation
 *  in column-major format.
 * @return Vector of feature vectors with class = 1 or class = 0
 */
Matrix getTestDataCm() {
    Matrix matRm = getTestDataRm();
    int numCols = matRm[0].size();
    int numRows = matRm.size();
    Matrix matCm;
    std::vector<double> col(numRows);

    for (int i = 0; i < numCols; i++){
        for (int j = 0; j < numRows; j++){
            col[j] = matRm[j][i];
        }
        matCm.emplace_back(col);
    }
    return matCm;
}


TEST_CASE("Test binary logistic regression fitting row-major", "[classic]") {
    int numFeatures = 3;

    Matrix mat = getTestDataRm();

    std::vector<std::vector<double>> X(mat.size());
    std::vector<double> Y(mat.size());

    for (size_t i = 0; i < mat.size(); i++) {
        X[i] = std::vector<double>(mat[i].begin(), mat[i].begin() + numFeatures);
        Y[i] = mat[i][numFeatures];
    }

    X = ml4cpp::LinearAlgebra::normaliseDataRm(X);
    ml4cpp::BinaryLogisticRegressorRm<double> model(numFeatures);
    model.fit(X, Y);
    model.evaluate(X, Y);

    for (double coefficient: model.getCoefficients()){
        REQUIRE(coefficient != 0);
    }

}


TEST_CASE("Test binary logistic regression fitting column-major", "[classic]") {
    int numFeatures = 3;
    Matrix mat = getTestDataCm();
    std::vector<std::vector<double>> X(numFeatures);
    std::vector<double> Y = mat[numFeatures];

    for (int i = 0; i < numFeatures; i++){
        X[i] = mat[i];
    }

    X = ml4cpp::LinearAlgebra::normaliseDataCm(X);
    ml4cpp::BinaryLogisticRegressorCm<double> model(numFeatures);
    model.fit(X, Y);
    model.evaluate(X, Y);

    for (double coefficient: model.getCoefficients()){
        REQUIRE(coefficient != 0);
    }
}

TEST_CASE("Test sigmoid function", "[classic]") {
    ml4cpp::AbstractBinaryLogisticRegressor<double> model;

    REQUIRE(model.sigmoid(100) == 100 / (1.0 + std::fabs(100)));
    REQUIRE(model.sigmoid(0.222) == 0.222 / (1.0 + std::fabs(0.222)));
    REQUIRE(model.sigmoid(-1.222) == -1.222 / (1.0 + std::fabs(-1.222)));
}