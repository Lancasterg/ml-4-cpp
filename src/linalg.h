#ifndef LINALG_H_
#define LINALG_H_

#include<vector>
#include<iostream>
#include<cstdio>

#define Matrix vector<vector<double>>

using namespace std;

class LinearAlgebra
{
private:
	void printColumnVector(vector<double> vec);

public:

	// Constructor
    LinearAlgebra();

    vector<double> addVectors(vector<double> a, vector<double> b);
    vector<double> scalarMult(double s, vector<double> v);

    double dotProd(vector<double> a, vector<double> b);

    Matrix scalarMatMult(double s, Matrix m);

    Matrix matMult(Matrix a, Matrix b);
    Matrix transpose(Matrix mat);

    void printVector(vector<double> vec);
    void printMatrix(Matrix mat);

};

#endif

