#include "../libs/catch.hpp"
#include "../linear_regression/MultipleLinearRegressor.h"
#include <../linear_regression/MultipleLinearRegressor.cpp> // required for templating
#include "../linear_regression/AbstractRegressorModel.cpp" // required for templating
#include <FileReader.h>





TEST_CASE("Test multiple predictions", "[classic]") {
    ml4cpp::MultipleLinearRegressor<double> model;
    model.setCoefficients(10, {1, 2});
    REQUIRE(model.predict({1, 10}) == 31);

}


TEST_CASE("Test multiple regression fitting", "[classic]") {
    std::string test_path = "/Users/george/eclipse-workspace/ml-4-cpp/data/multiple_linear_regression_data.csv";
    ml4cpp::FileReader fileReader;
    int numFeatures = 4;

    Matrix mat = fileReader.readCsvRm(test_path, numFeatures + 1);

    std::vector<std::vector<double>> X(mat.size());
    std::vector<double> Y(mat.size());

    for (size_t i = 0; i < mat.size(); i++) {
        X[i] = std::vector<double>(mat[i].begin(), mat[i].begin() + numFeatures);
        Y[i] = mat[i][numFeatures];
    }

    X = ml4cpp::LinearAlgebra::normaliseDataRm(X);

    ml4cpp::MultipleLinearRegressor<double> model(numFeatures);

    model.fit(X, Y);

    std::cout << "mse: " << model.meanSquaredError(X, Y) << std::endl;

}

