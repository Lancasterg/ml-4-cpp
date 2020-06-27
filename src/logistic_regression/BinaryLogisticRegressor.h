#ifndef ML_4_CPP_BINARYLOGISTICREGRESSOR_H
#define ML_4_CPP_BINARYLOGISTICREGRESSOR_H


#include <vector>
#include <string>

namespace ml4cpp {

    template<class T>
    class AbstractBinaryLogisticRegressor {

    protected:
        std::vector<double> coefficients;
        double bias = 0;
        double threshold = 0;
        int num_coefficients= 0;

    public:

        std::vector<double> getCoefficients();

        double sigmoid(T x);

        double predictProba(std::vector<T> x);

        int predict(std::vector<T> x);

        virtual void fit(std::vector<std::vector<T>> X, std::vector<T> Y);

        virtual void evaluate(std::vector<std::vector<T>> X, std::vector<T> Y);
    };

    template <class T>
    class BinaryLogisticRegressorCm : public AbstractBinaryLogisticRegressor<T> {


    public:

        explicit BinaryLogisticRegressorCm(int n_features);

        void fit(std::vector<std::vector<T>> X, std::vector<T> Y) override;

        double crossEntropy(std::vector<std::vector<T>> X, std::vector<T> Y);

        void evaluate(std::vector<std::vector<T>> X, std::vector<T> Y) override;

    };

    template<class T>
    class BinaryLogisticRegressorRm : public AbstractBinaryLogisticRegressor<T> {

    public:
        BinaryLogisticRegressorRm() = default;

        explicit BinaryLogisticRegressorRm(int n_features);

        void fit(std::vector<std::vector<T>> X, std::vector<T> Y) override;

        double crossEntropy(std::vector<std::vector<T>> X, std::vector<T> Y);

        void evaluate(std::vector<std::vector<T>> X, std::vector<T> Y) override;

    };
}


#endif //ML_4_CPP_BINARYLOGISTICREGRESSOR_H
