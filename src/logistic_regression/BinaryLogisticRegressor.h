#ifndef ML_4_CPP_BINARYLOGISTICREGRESSOR_H
#define ML_4_CPP_BINARYLOGISTICREGRESSOR_H


#include <vector>
#include <string>

namespace ml4cpp {


    class AbstractBinaryLogisticRegressor {

    protected:
        std::vector<double> coefficients;
        double bias;
        double threshold;
        int num_coefficients;

    public:

        double sigmoid(double x);

        double predictProba(std::vector<double> x);

        int predict(std::vector<double> x);

        virtual void fit(std::vector<std::vector<double>> X, std::vector<double> Y);

        virtual void evaluate(std::vector<std::vector<double>> X, std::vector<double> Y);
    };


    class BinaryLogisticRegressorCm : public AbstractBinaryLogisticRegressor {


    public:

        explicit BinaryLogisticRegressorCm(int n_features);

        void fit(std::vector<std::vector<double>> X, std::vector<double> Y) override;

        double crossEntropy(std::vector<std::vector<double>> X, std::vector<double> Y);

        void evaluate(std::vector<std::vector<double>> X, std::vector<double> Y) override;

    };


    class BinaryLogisticRegressorRm : public AbstractBinaryLogisticRegressor {

    public:
        BinaryLogisticRegressorRm() = default;

        explicit BinaryLogisticRegressorRm(int n_features);

        void fit(std::vector<std::vector<double>> X, std::vector<double> Y) override;

        double crossEntropy(std::vector<std::vector<double>> X, std::vector<double> Y);

        void evaluate(std::vector<std::vector<double>> X, std::vector<double> Y) override;

    };
}


#endif //ML_4_CPP_BINARYLOGISTICREGRESSOR_H
