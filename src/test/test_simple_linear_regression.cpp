#include "../libs/catch.hpp"
#include "../linear_regression/SimpleLinearRegressor.h"
#include "../linear_regression/SimpleLinearRegressor.cpp"
#include "../linear_regression/AbstractRegressorModel.cpp"



/**
 * Return some simple training data to test fitting and evaluation.
 * @return Feature vector
 */
std::vector<double> getTestX() {
    std::vector<double> X(1000);
    for (size_t i = 0; i < 1000; i++) {
        X[i] = (double) i / 10.0;
    }
    return X;
}

/**
 * Return some simple training label to test fitting and evaluation.
 * @param X: Input features
 * @return Feature vector
 */
std::vector<double> getTestY(std::vector<double>X){
    std::vector<double> Y(X.size());

    for (size_t i = 0; i < X.size(); i++){
        Y[i] = 100 + (3 * X[i]);
    }
    return Y;
}



TEST_CASE("Test simple fitting", "[classic]") {
    std::vector<double> X = getTestX();
    std::vector<double> Y = getTestY(X);

    ml4cpp::SimpleLinearRegressor<double> model(0);
    model.fit(X, Y, 0.0001, 1000);

    std::cout << "" << "";



}

TEST_CASE("Test simple predictions", "[classic]") {
    ml4cpp::SimpleLinearRegressor<double> model(0);

    // Test 1
    model.setCoefficients({1, 2});
    REQUIRE(model.predict(10) == 21);

    // Test 2
    model.setCoefficients({100, 2000});
    REQUIRE(model.predict(0.5) == 1100);

    // Test 3
    model.setCoefficients({10.2, 0.1});
    REQUIRE(model.predict(24.2) == 12.62);

}