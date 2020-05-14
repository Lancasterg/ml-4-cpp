//
// Created by George Lancaster on 14/05/2020.
//

#include <vector>
#include <set>
#include <cmath>
#include "DecisionTreeClassifier.h"

namespace ml4cpp {
/**
 * DecisionTreeClassifier code
*/

    /**
     * Train the decision tree
     * @param X
     * @param Y
     */
    void DecisionTreeClassifier::fit(std::vector<std::vector<double>> X, std::vector<double> Y) {


    }

    int DecisionTreeClassifier::predict(std::vector<double> x) {

    }


    void predict(std::vector<double> x) {

    }


    /**
 * DecisionTreeNode code
 */


    DecisionTreeNode *DecisionTreeNode::getLeftChild() {
        return leftChild;
    }

    DecisionTreeNode *DecisionTreeNode::getRightChild() {
        return rightChild;
    }

    void DecisionTreeNode::createLeftChild() {

        leftChild = new DecisionTreeNode();
        leftChild->setParent(this);

    }

    void DecisionTreeNode::createRightChild() {
        rightChild = new DecisionTreeNode();
        rightChild->setParent(this);

    }

    void DecisionTreeNode::createChild() {
        if (leftChild == nullptr) {
            createLeftChild();
        } else if (rightChild == nullptr) {
            createRightChild();
        }
    }

    void DecisionTreeNode::setParent(DecisionTreeNode *parentNode) {
        parent = parentNode;
    }

    double DecisionTreeNode::giniImpurity(std::vector<std::vector<double>> X, std::vector<double> Y) {
        double impurity = 1;
        int uniqueCount = (int) std::set<int> (Y.begin(), Y.end()).size();

        std::vector<int> classCount(uniqueCount, 0);

        for (double y : Y) {
            classCount[y]++;
        }

        for (int nClass : classCount){
            impurity -= std::pow(nClass / (double) Y.size(), 2);
        }
        return impurity;
    }
}

