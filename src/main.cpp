#include <iostream>
#include <chrono>
#include "logistic_regression/BinaryLogisticRegressor.h"
#include "logistic_regression/BinaryLogisticRegressor.cpp"
#include "linear_algebra/linalg.h"
#include "linear_regression/AbstractRegressorModel.h"
#include "linear_regression/AbstractRegressorModel.cpp"
#include "linear_regression/SimpleLinearRegressor.h"
#include "linear_regression/MultipleLinearRegressor.h"
#include "linear_regression/SimpleLinearRegressor.cpp"
#include "linear_regression/MultipleLinearRegressor.cpp"
#include "helpers/FileReader.h"

using namespace std;
using namespace std::chrono;
using namespace ml4cpp;


void runSimpleLinearRegression(Matrix mat){
	vector<double> X = mat[0];
	vector<double> Y = mat[1];

	LinearAlgebra linalg;
	SimpleLinearRegressor<double> model(1);

	auto start = high_resolution_clock::now();
	model.fit(X, Y, 0.0000001, 100000);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	cout << "Time taken by function: " << duration.count() / 1000 << " ms" << endl;
	linalg.printVector(model.getCoefficients());

}


void runMultipleLinearRegression(Matrix mat){
	Matrix X;
	vector<double> Y;
	LinearAlgebra linalg;
	MultipleLinearRegressor<double> model(mat.size());

	// Create feature matrix
	// Put 1 in first column to
	// allow for vector multiplication
	vector<double> firstColumn(mat[0].size(), 1);
	X.push_back(firstColumn);
	for (int i = 0; i < mat.size() -1; i++){
		X.push_back(mat[i]);
	}

	// Create target vector
	Y = mat[mat.size()-1];
	model.fit(X, Y);

}

void runLogisticRegression(Matrix mat){
    int numFeatures = 3;

    std::vector<std::vector<double>> X(mat.size());
    std::vector<double> Y(mat.size());

    for (size_t i = 0; i < mat.size(); i++) {
        X[i] = std::vector<double>(mat[i].begin(), mat[i].begin() + numFeatures);
        Y[i] = mat[i][numFeatures];
    }

    X = ml4cpp::LinearAlgebra::normaliseDataRm(X);
    BinaryLogisticRegressorRm<double> model(numFeatures);
    model.fit(X, Y);
    model.evaluate(X, Y);


}

int main(int argc, char** argv) {

    if (argc != 3){
        throw runtime_error("Must specify input data location.");
    }

	string file = argv[1];
    string algorithm = argv[2];
	int ncols = atoi(argv[3]);

    FileReader fileReader;
	Matrix mat = fileReader.readCsvRm(file, ncols);

	// Multiple linear regression
	if (algorithm == "mlr"){
        runMultipleLinearRegression(mat);
    }
	// Simple linear regression
	else if (algorithm == "slr"){
	    runSimpleLinearRegression(mat);
	}
	// Logistic regression
	else if (algorithm == "lr"){
	    runLogisticRegression(mat);
	}

	return 0;
}
