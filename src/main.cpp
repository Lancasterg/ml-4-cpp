#include <iostream>
#include <chrono>
#include "linear_algebra/linalg.h"
#include "helpers/csv.h"
#include "linear_regression/SimpleLinearRegressor.h"
#include "linear_regression/MultipleLinearRegressor.h"

using namespace std;
using namespace std::chrono;
using namespace ml4cpp;

Matrix readSimpleCsv(string file){
	io::CSVReader<2> in(file);
	double x; double y;
	Matrix ret(2);

	in.read_header(io::ignore_extra_column, "X", "Y");
	while(in.read_row(x, y)){
	    ret[0].push_back(x);
	    ret[1].push_back(y);
	}
	return ret;
}


Matrix readMultipleCsv(string file){
	io::CSVReader<5> in(file);
	double a, b, c, d, y;
	vector<double> cool;

	Matrix ret(5);

	in.read_header(io::ignore_extra_column, "A", "B", "C", "D", "Y");

	while(in.read_row(a, b, c, d, y)){
	    ret[0].push_back(a);
	    ret[1].push_back(b);
	    ret[2].push_back(c);
	    ret[3].push_back(d);
	    ret[4].push_back(y);

	}
	return ret;

}

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
	string file = argv[1];
	Matrix mat = readMultipleCsv(file);

	runMultipleLinearRegression(mat);


	return 0;
}
