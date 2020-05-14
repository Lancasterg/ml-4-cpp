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

    public:
        DecisionTreeNode() = default;

        DecisionTreeNode *getLeftChild();

        DecisionTreeNode *getRightChild();

        void createChild();

        void createLeftChild();

        void createRightChild();

        void setParent(DecisionTreeNode *parentNode);

        short int question();

        double giniImpurity(std::vector<std::vector<double>> X, std::vector<double> Y);
    };


    class DecisionTreeClassifier {

    private:
        DecisionTreeNode *rootNode;


    public:
        DecisionTreeClassifier() = default;

        void fit(std::vector<std::vector<double>> X, std::vector<double> Y);
        int predict(std::vector<double> x);


        void fit(std::vector<std::vector<double>> X);
    };

}

#endif //ML_4_CPP_DECISIONTREECLASSIFIER_H
