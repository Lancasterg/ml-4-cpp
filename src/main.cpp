#include <iostream>
#include <chrono>
#include "linear_algebra/linalg.h"
#include "linear_regression/SimpleLinearRegressor.h"
#include "linear_regression/MultipleLinearRegressor.h"
#include "helpers/FileReader.h"

using namespace std;
using namespace std::chrono;
using namespace ml4cpp;


void runSimpleLinearRegression(Matrix mat){
	vector<double> X = mat[0];
	vector<double> Y = mat[1];

	LinearAlgebra linalg;
	SimpleLinearRegressor model(1);

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
	MultipleLinearRegressor model(mat.size());

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

int main(int argc, char** argv) {

    if (argc != 2){
        throw runtime_error("Must specify input data location.");
    }

    FileReader fileReader;
	string file = argv[1];

//	string file = "/Users/george/eclipse-workspace/ml-4-cpp/data/multiple_linear_regression_data.csv";
	Matrix mat = fileReader.readMultipleCsv(file);
	runMultipleLinearRegression(mat);


	return 0;
}
