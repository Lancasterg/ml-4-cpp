#include "../libs/catch.hpp"
#include "../linear_algebra/linalg.h"
#include <vector>

/**
 * Compare the values of two vectors.
 * @param result: Vector result of test function
 * @param truth: Vector containing true values
 */
void compareVectors(std::vector<double> &result, std::vector<double> &truth) {
    for (int i = 0; i < result.size(); i++) {
        REQUIRE(result[i] == truth[i]);
    }
}

/**
 * Compare the values of two vectors.
 * @param result: Matrix result of test function
 * @param truth: Matrix containing true values
 */
void compareMatrices(Matrix &result, Matrix &truth) {
    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < result[i].size(); j++) {
            REQUIRE(truth[i][j] == result[i][j]);
        }
    }

}

/**
 * Test scalar multiplication with a vector.
 * @param vec: Test vector
 * @param scalar: Test scalar
 * @param truth: Truth of vector times scalar
 */
void testScalarMultiplication(std::vector<double> vec, double scalar, std::vector<double> truth) {
    std::vector<double> result = ml4cpp::LinearAlgebra::scalarMult(scalar, vec);
    compareVectors(result, truth);
}

/**
 * Test addition of two vectors.
 * @param test_a
 * @param test_b
 * @param truth
 */
void testVectorAddition(std::vector<double> test_a, std::vector<double> test_b, std::vector<double> truth) {
    std::vector<double> result = ml4cpp::LinearAlgebra::addVectors(test_a, test_b);
    compareVectors(result, truth);
}

/**
 * Test the dot product of two two vectors.
 * @param vec_a: First vector to be used in dot prod
 * @param vec_b: Second vector to be used in dot prod
 * @param truth: The true result of the dot product of vec_a and vec_b
 */
void testDotProduct(std::vector<double> vec_a, std::vector<double> vec_b, double truth) {
    double result = ml4cpp::LinearAlgebra::dotProd(vec_a, vec_b);
    REQUIRE(result == truth);
}

/**
 * Test matrix multiplication by a scalar.
 * @param s: Scalar by which to multiply m
 * @param m: Matrix to be multiplied by m
 * @param truth: Truth matrix sm
 */
void testScalarMatMult(double s, Matrix m, Matrix truth) {
    Matrix result = ml4cpp::LinearAlgebra::scalarMatMult(s, m);
    compareMatrices(result, truth);
}

/**
 * Test matrix, matrix multiplication.
 * @param a: First matrix used in product
 * @param b: Second matrix used in product
 * @param truth: Truth matrix ab
 */
void testMatMultSquare(Matrix a, Matrix b, Matrix truth) {
    Matrix result = ml4cpp::LinearAlgebra::matMultSquare(a, b);
    compareMatrices(result, truth);
}

TEST_CASE("Test vector addition", "[classic]") {

    // Test 1
    testVectorAddition({1, 2, 3, 4},
                       {9, 10, 20, 5},
                       {10, 12, 23, 9});

    // Test 2
    testVectorAddition({100, 300, 999, 1},
                       {200, 10, 1, 5},
                       {300, 310, 1000, 6});

    // Test 3
    testVectorAddition({0.5555555, 100000, 0.9, 10, 50},
                       {0.1000000, 100000, 1.0, 50, 50},
                       {0.6555555, 200000, 1.9, 60, 100});
}

TEST_CASE("Test scalar multiplication", "[classic]") {

    // Test 1
    testScalarMultiplication({1, 2, 3, 4}, 0.5, {0.5, 1, 1.5, 2});

    // Test 2
    testScalarMultiplication({0, 0}, 1000, {0, 0});

    // Test 3
    testScalarMultiplication({10000.53453, 16723.82, 236.93}, 10.1,
                             {101005.398753, 168910.582, 2392.993});
}

TEST_CASE("Test dot product", "[classic]") {

    // Test 1
    testDotProduct({1, 2, 3}, {1, 5, 7}, 32);

    // Test 2
    testDotProduct({100, 2.5, 2}, {100.1, 3, 3}, 10023.5);

    // Test 3
    testDotProduct({10, 0.1, 0.1}, {10, 0.1, 1}, 100.11);


}

TEST_CASE("Test matrix scalar multiplication", "[classic]") {

    // Test 1
    testScalarMatMult(10, {{10, 20},
                           {30, 40}}, {{100, 200},
                                       {300, 400}});

    // Test 2
    testScalarMatMult(0.5, {{10, 20},
                            {20, 10}}, {{5,  10},
                                        {10, 5}});

    // Test 3
    testScalarMatMult(0.1, {{100,    250},
                            {100000, 0.5}}, {{10,    25},
                                             {10000, 0.05}});

}

TEST_CASE("Test matrix, matrix multiplication", "[classic]") {


    // Test 1
    testMatMultSquare({{1, 2},
                       {3, 4}}, {{5, 6},
                                 {7, 8}}, {{19, 22},
                                           {43, 50}});

    // Test 2
    testMatMultSquare({{0.5, 10},
                       {22,  3}}, {{0.9, 0.9},
                                   {1,   7000}}, {{10.45, 70000.45},
                                                  {22.8,  21019.8}});

    // Test 3
    testMatMultSquare({{1, 0},
                       {0, 1}}, {{50, 60},
                                 {70, 80}}, {{50, 60},
                                             {70, 80}});


}

TEST_CASE("Test", "[classic]") {
    REQUIRE(1 == 1);
}