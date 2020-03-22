#ifndef LINALG_H_
#define LINALG_H_

#include <vector>
#include <iostream>
#include <cstdio>

#define Matrix std::vector<std::vector<double>>

namespace ml4cpp{

	class LinearAlgebra
	{
	private:
		void printColumnVector(std::vector<double> vec);

	public:

		// Constructor
		LinearAlgebra();

		std::vector<double> addVectors(std::vector<double> a, std::vector<double> b);
		std::vector<double> scalarMult(double s, std::vector<double> v);

		double dotProd(std::vector<double> a, std::vector<double> b);

		Matrix scalarMatMult(double s, Matrix m);

		Matrix matMult(Matrix a, Matrix b);
		Matrix transpose(Matrix mat);

		void printVector(std::vector<double> vec);
		void printMatrix(Matrix mat);

	};
};
#endif

