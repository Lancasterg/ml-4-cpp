if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug)
endif()

cmake_minimum_required(VERSION 3.16)
project(ml_4_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(src/helpers
        src/linear_algebra
        src/linear_regression
        src/logistic_regression)

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
        src/logistic_regression/BinaryLogisticRegressor.cpp
        src/logistic_regression/BinaryLogisticRegressor.h
        src/decision_tree/DecisionTreeClassifier.cpp
        src/decision_tree/DecisionTreeClassifier.h src/py_bindings.cpp)

add_executable(
        tests
        src/libs/catch.hpp
        src/test/test_runner.cpp
        src/test/test_linalg.cpp
        src/test/test_simple_linear_regression.cpp
        src/test/test_multiple_linear_regression.cpp
        src/test/test_binary_logistic_regression.cpp

        src/linear_algebra/linalg.cpp
        src/linear_algebra/linalg.h

        src/linear_regression/AbstractRegressorModel.cpp
        src/linear_regression/AbstractRegressorModel.h
        src/linear_regression/MultipleLinearRegressor.cpp
        src/linear_regression/MultipleLinearRegressor.h
        src/linear_regression/SimpleLinearRegressor.cpp
        src/linear_regression/SimpleLinearRegressor.h
        src/logistic_regression/BinaryLogisticRegressor.cpp
        src/logistic_regression/BinaryLogisticRegressor.h
        src/decision_tree/DecisionTreeClassifier.cpp
        src/decision_tree/DecisionTreeClassifier.h

        src/helpers/FileReader.cpp
        src/helpers/FileReader.h
        src/libs/csv.h
        src/test/test_decision_tree_classifier.cpp
)
