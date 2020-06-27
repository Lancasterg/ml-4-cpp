#include "linalg.h"
#include <numeric>
#include <cmath>

namespace ml4cpp {


    LinearAlgebra::LinearAlgebra() = default;


    /**
     * Return the sum of two vectors.
     * @param a: Vector a
     * @param b: Vector b
     * @return The sum of a and b
     */
    std::vector<double> LinearAlgebra::addVectors(std::vector<double> a, std::vector<double> b) {

        if (a.size() != b.size()) {
            printf("Vectors must be the same length");
            exit(-1);
        }

        for (size_t i = 0; i < a.size(); i++) {
            a[i] += b[i];
        }

        return a;

    }

    /**
     * Multiply a matrix by a scalar value
     * @param s: The scalar to multiply by
     * @param m: The matrix to be multiplied
     * @return
     */
    Matrix LinearAlgebra::scalarMatMult(double s, Matrix m) {
        for (size_t i = 0; i < m.size(); i++) {
            for (size_t j = 0; j < m[0].size(); j++) {
                m[i][j] *= s;
            }
        }

        return m;
    }


    /**
     * Return the product of two matrices
     * @param a: Matrix a
     * @param b: Matrix b
     * @return product of a and b
     */
    Matrix LinearAlgebra::matMultSquare(Matrix a, Matrix b) {

        if (a[1].size() != b.size()) {
            printf("Vectors must be the same length");
            exit(-1);
        }

        // Instantiate result matrix
        std::vector<std::vector<double>> mat(a.size());

        for (size_t i = 0; i < b[1].size(); i++) {
            mat[i].resize(b[1].size());
        }

        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                for (size_t k = 0; k < a[0].size(); ++k) {
                    mat[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return mat;
    }


    /**
     * Return the dot product of two vectors
     * @param a: vector a
     * @param b: vector b
     * @return Dot product of a and b
     */
    double LinearAlgebra::dotProd(std::vector<double> a, std::vector<double> b) {

        if (a.size() != b.size()) {
            printf("Vectors must be the same length");
            exit(-1);
        }

        double ret = 0;

        for (size_t i = 0; i < a.size(); i++) {
            ret += a[i] * b[i];
        }

        return ret;
    }


    /**
     * Print a column vector
     * @param vec: The vector to be printed
     */
    void LinearAlgebra::printColumnVector(std::vector<double> vec) {

        for (size_t i = 0; i < vec.size(); i++) {
            if (i < vec.size() - 1) {
                printf("%f, ", vec[i]);
            } else {
                printf("%f", vec[i]);
            }
        }
    }

    /**
     * Pretty-print a column vector
     * @param vec: The vector to be printed
     */
    void LinearAlgebra::printVector(std::vector<double> vec) {
        printf("[");
        printColumnVector(vec);
        printf("]");
    }

    /**
     * Pretty-print a matrix
     * @param mat: The matrix to be printed
     */
    void LinearAlgebra::printMatrix(Matrix mat) {
        printf("[");
        for (size_t i = 0; i < mat.size(); i++) {

            printVector(mat[i]);

            if (i < mat[i].size() - 2) {
                printf("\n");
            }
        }
        printf("]\n");
    }

    /**
     * Multiply a vector by a scalar
     * @param s: The scalar to multiply by
     * @param v: The vector to be multiplied
     * @return The product of s and v
     */
    std::vector<double> LinearAlgebra::scalarMult(double s, std::vector<double> v) {
        for (double &i : v) {
            i *= s;
        }
        return v;
    }

    Matrix LinearAlgebra::transpose(Matrix mat) {
        return mat;
    }

    Matrix LinearAlgebra::normaliseDataCm(std::vector<std::vector<double>> mat) {
        std::vector<double> means(mat.size());
        std::vector<double> std_devs(mat.size());

        for (size_t i = 0; i < mat.size(); i++){
            means[i] = std::accumulate(mat[i].begin(), mat[i].end(), 0.0);
            means[i] /= mat[i].size();

            double sq_sum = std::inner_product(mat[i].begin(), mat[i].end(), mat[i].begin(), 0.0);
            std_devs[i] = std::sqrt(sq_sum / mat[i].size() - means[i] * means[i]);
        }

        for (size_t i = 0; i < mat.size(); i++){
            for (size_t j = 0; j < mat[i].size(); j++){
                mat[i][j] = (mat[i][j] - means[i]) / std_devs[i];
            }
        }

        return mat;
    }


    Matrix LinearAlgebra::normaliseDataRm(std::vector<std::vector<double>> mat) {
        int n_features = mat[0].size();
        int n_rows = mat.size();
        std::vector<double> means(n_features);
        std::vector<double> std_devs(n_features);

        for (int i = 0; i < n_rows; i++){
            for (int j = 0; j < n_features; j++){
                means[j] += mat[i][j];
            }
        }
        for (int j = 0; j < n_features; j++){
            means[j] /= mat.size();
        }
        for (int i = 0; i < n_rows; i++){
            for (int j = 0; j < n_features; j++){
                std_devs[j] += pow(mat[i][j] - means[j], 2);
            }
        }
        for (int j = 0; j < n_features; j++){
            std_devs[j] = sqrt(std_devs[j] /= mat.size());
        }
        for (int i = 0; i < n_rows; i++){
            for (int j = 0; j < n_features; j++){
                mat[i][j] = (mat[i][j] - means[j]) / std_devs[j];
            }
        }
        return mat;
    }




};
