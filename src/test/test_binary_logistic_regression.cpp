#include <FileReader.h>
#include "../libs/catch.hpp"
#include "../logistic_regression/BinaryLogisticRegressor.h"


TEST_CASE("Test binary logistic regression fitting row-major", "[classic]") {
    std::string test_path = "/Users/george.lancaster/Projects/learning/cpp/ml-4-cpp/data/classification_data.csv";
    ml4cpp::FileReader fileReader;
    int numCols = 4;
    int numFeatures = 3;

    Matrix mat = fileReader.readCsvRm(test_path, numCols);
    std::vector<std::vector<double>> X(mat.size());
    std::vector<double> Y(mat.size());

    for (size_t i = 0; i < mat.size(); i++) {
        X[i] = std::vector<double>(mat[i].begin(), mat[i].begin() + numFeatures);
        Y[i] = mat[i][numFeatures];
    }

    X = ml4cpp::LinearAlgebra::normaliseDataRm(X);
    ml4cpp::BinaryLogisticRegressorRm model(numFeatures);
    model.fit(X, Y);
    model.evaluate(X, Y);
}


TEST_CASE("Test binary logistic regression fitting column-major", "[classic]") {
    std::string test_path = "/Users/george.lancaster/Projects/learning/cpp/ml-4-cpp/data/classification_data.csv";
    ml4cpp::FileReader fileReader;
    int numCols = 4;
    int numFeatures = 3;

    Matrix mat = fileReader.readCsvCm(test_path, numCols);
    std::vector<std::vector<double>> X(numFeatures);
    std::vector<double> Y = mat[numFeatures];

    for (int i = 0; i < numFeatures; i++) {
        X[i] = std::vector<double>(mat[i].begin(), mat[i].end());
    }

    X = ml4cpp::LinearAlgebra::normaliseDataCm(X);
    ml4cpp::BinaryLogisticRegressorCm model(numFeatures);
    model.fit(X, Y);
    model.evaluate(X, Y);
}