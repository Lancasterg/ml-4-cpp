#include "../libs/catch.hpp"
#include <MultipleLinearRegressor.h>
#include <FileReader.h>


TEST_CASE("Test multiple predictions", "[classic]") {
    ml4cpp::MultipleLinearRegressor model;

    // Test 1
    model.setCoefficients(10, {1, 2});
    REQUIRE(model.predict({1, 10}) == 31);

    // Test 2
//    model.setCoefficients({100, 2000, 400, 10});
//    REQUIRE(model.predict({1, 2, 3, 4}) == );
//
//    // Test 3
//    model.setCoefficients({10.2, 0.1});
//    REQUIRE(model.predict(24.2) == 12.62);

}


TEST_CASE("Test multiple regression fitting", "[classic]") {
    std::string test_path = "/Users/george/eclipse-workspace/ml-4-cpp/data/multiple_linear_regression_data.csv";
    ml4cpp::FileReader fileReader;

    Matrix X;
    Matrix mat = fileReader.readMultipleCsv(test_path);

    for (int i = 0; i < mat.size() - 1; i++) {
        X.push_back(mat[i]);
    }

    X = ml4cpp::LinearAlgebra::normaliseData(X);

    std::vector<double> Y = mat[mat.size() - 1];

    ml4cpp::MultipleLinearRegressor model(X.size());

    model.fit(X, Y);

    std::cout << model.meanSquaredError(X, Y);

}

