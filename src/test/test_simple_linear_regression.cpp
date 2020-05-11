#include "../libs/catch.hpp"
#include "../linear_regression/SimpleLinearRegressor.h"


TEST_CASE("Test simple predictions", "[classic]") {
    ml4cpp::SimpleLinearRegressor model;

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