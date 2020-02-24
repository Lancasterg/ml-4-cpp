#include <iostream>
#include <chrono>
#include "linear_algebra/linalg.h"
#include "linear_regression/LinearRegressor.h"
#include "helpers/csv.h"

using namespace std;
using namespace std::chrono;

Matrix readCsv(string file){
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

int main() {
	string file = "/Users/george/eclipse-workspace/ml-4-cpp/data/linear_regression_data.csv";
	Matrix df = readCsv(file);

	vector<double> X = df[0];
	vector<double> Y = df[1];

	LinearAlgebra linalg;
	LinearRegressor model(1);
	auto start = high_resolution_clock::now();
	model.fit(X, Y, 0.0000001, 100000);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time taken by function: " << duration.count() / 1000 << " ms" << endl;
//	printf("%d", model.)
	linalg.printVector(model.getCoefficients());


	return 0;
}
