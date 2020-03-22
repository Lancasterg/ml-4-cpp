#include <iostream>
#include <chrono>
#include "linear_algebra/linalg.h"
#include "helpers/csv.h"
#include "linear_regression/SimpleLinearRegressor.h"

using namespace std;
using namespace std::chrono;
using namespace ml4cpp;

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

int main(int argc, char** argv) {
	string file = argv[1];

	Matrix df = readCsv(file);

	vector<double> X = df[0];
	vector<double> Y = df[1];

	LinearAlgebra linalg;
	SimpleLinearRegressor model(1);
	auto start = high_resolution_clock::now();
	model.fit(X, Y, 0.0000001, 100000);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time taken by function: " << duration.count() / 1000 << " ms" << endl;
//	printf("%d", model.)
	linalg.printVector(model.getCoefficients());


	return 0;
}
