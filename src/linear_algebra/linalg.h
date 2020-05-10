#ifndef LINALG_H_
#define LINALG_H_

#include <vector>
#include <iostream>
#include <cstdio>

#define Matrix std::vector<std::vector<double>>

namespace ml4cpp {

    class LinearAlgebra {
    private:
        static void printColumnVector(std::vector<double> vec);

    public:

        // Constructor
        LinearAlgebra();

        static std::vector<double> addVectors(std::vector<double> a, std::vector<double> b);

        static std::vector<double> scalarMult(double s, std::vector<double> v);

        static double dotProd(std::vector<double> a, std::vector<double> b);

        static Matrix scalarMatMult(double s, Matrix m);

        static Matrix matMult(Matrix a, Matrix b);

        static Matrix transpose(Matrix mat);

        static void printVector(std::vector<double> vec);

        void printMatrix(Matrix mat);

    };
};
#endif

