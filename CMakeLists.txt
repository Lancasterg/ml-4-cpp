cmake_minimum_required(VERSION 3.16)
project(ml_4_cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/helpers
        src/linear_algebra
        src/linear_regression)

add_executable(
        ml_4_cpp
        src/libs/csv.h
        src/linear_algebra/linalg.cpp
        src/linear_algebra/linalg.h
        src/linear_regression/AbstractRegressorModel.cpp
        src/linear_regression/AbstractRegressorModel.h
        src/linear_regression/MultipleLinearRegressor.cpp
        src/linear_regression/MultipleLinearRegressor.h
        src/linear_regression/SimpleLinearRegressor.cpp
        src/linear_regression/SimpleLinearRegressor.h
        src/main.cpp
        src/helpers/FileReader.cpp
        src/helpers/FileReader.h
        )

add_executable(
        tests
        src/libs/catch.hpp
        src/test/test_runner.cpp
        src/test/test_linalg.cpp
        src/test/test_simple_linear_regression.cpp
        src/test/test_multiple_linear_regression.cpp
        src/linear_algebra/linalg.cpp
        src/linear_algebra/linalg.h
        src/linear_regression/AbstractRegressorModel.cpp
        src/linear_regression/AbstractRegressorModel.h
        src/linear_regression/MultipleLinearRegressor.cpp
        src/linear_regression/MultipleLinearRegressor.h
        src/linear_regression/SimpleLinearRegressor.cpp
        src/linear_regression/SimpleLinearRegressor.h
        src/helpers/FileReader.cpp
        src/helpers/FileReader.h
        src/libs/csv.h
        )
