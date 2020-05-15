#ifndef ML_4_CPP_DECISIONTREECLASSIFIER_H
#define ML_4_CPP_DECISIONTREECLASSIFIER_H

#include <string>

namespace ml4cpp {


    class DecisionTreeNode {
    private:
        ml4cpp::DecisionTreeNode *parent = nullptr;
        ml4cpp::DecisionTreeNode *leftChild = nullptr;
        ml4cpp::DecisionTreeNode *rightChild = nullptr;
        double value = 0;
        int column = 0;
        int prediction;

    public:
        bool leaf = false;

        DecisionTreeNode() = default;

        void setValue(double val);

        void setColumn(int col);

        DecisionTreeNode *getLeftChild();

        DecisionTreeNode *getRightChild();

        void createChild();

        void createLeftChild();

        void createRightChild();

        void setParent(DecisionTreeNode *parentNode);

        double giniImpurity(std::vector<std::vector<double>> X, std::vector<double> Y);

        short query(std::vector<double> x) const;


        std::tuple<std::vector<std::vector<double>>, std::vector<double>, std::vector<std::vector<double>>, std::vector<double>>
        partition(const std::vector<std::vector<double>> &X, const std::vector<double> &Y);

        double informationGain(std::vector<std::vector<double>> part_trueX, std::vector<double> part_trueY,
                               std::vector<std::vector<double>> part_falseX, std::vector<double> part_falseY,
                               double uncertainty);

        std::tuple<double, std::pair<int, double>> findBestSplit(std::vector<std::vector<double>> X, std::vector<double> Y);

        void setPrediction(int pred);

        int getPrediction();
    };


    class DecisionTreeClassifier {

    private:
        DecisionTreeNode *rootNode = new DecisionTreeNode();
        DecisionTreeNode *currentNode = rootNode;


    public:
        DecisionTreeClassifier() = default;

        void fit(std::vector<std::vector<double>> X, std::vector<double> Y);
        int predict(std::vector<double> x);

        int recursivePredict(std::vector<double> x);
    };

}

#endif //ML_4_CPP_DECISIONTREECLASSIFIER_H
