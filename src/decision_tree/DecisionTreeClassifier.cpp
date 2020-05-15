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
     * @param X: Feature vectors
     * @param Y: Target classifications
     */
    void DecisionTreeClassifier::fit(std::vector<std::vector<double>> X, std::vector<double> Y) {
        std::tuple<double, std::pair<int, double>> bestSplit;
        std::tuple<std::vector<std::vector<double>>,
                std::vector<double>,
                std::vector<std::vector<double>>,
                std::vector<double>> partitionedXY;


        bestSplit = currentNode->findBestSplit(X, Y);

        std::set<double> unique(Y.begin(), Y.end());
        if(std::get<0>(bestSplit) == 0 || unique.size() == 1) {
            // x[column] >= value;

            currentNode->leaf = true;
            currentNode->setPrediction(*unique.begin());
            return;
        }

        partitionedXY = currentNode->partition(X,Y);

        // fit true branch
        DecisionTreeNode *tmp_node = currentNode;
        currentNode->createRightChild();
        currentNode = currentNode->getRightChild();
        fit(std::get<0>(partitionedXY), std::get<1>(partitionedXY));

        // fit false branch
        tmp_node->createLeftChild();
        currentNode = tmp_node->getLeftChild();
        fit(std::get<2>(partitionedXY), std::get<3>(partitionedXY));

        currentNode = rootNode;
    }

    /**
     * Make a prediction and set the currentNode back to the root node so that more predictions can be made.
     * @param x: Feature vector
     * @return Classification of feature vector
     */
    int DecisionTreeClassifier::predict(std::vector<double> x){
        int classification = recursivePredict(x);
        currentNode = rootNode;
        return classification;

    }


    /**
     * Recursively traverse the tree structure to make a prediction.
     * @param x: Input feature vector
     * @return Classification of feature vector
     */
    int DecisionTreeClassifier::recursivePredict(std::vector<double> x) {
        if (currentNode->leaf){
            return currentNode->getPrediction();
        }

        DecisionTreeNode *tmp_node = currentNode;

        if (currentNode->query(x)){
            currentNode = tmp_node->getRightChild();
            return recursivePredict(x);
        }else{
            currentNode = tmp_node->getLeftChild();
            return recursivePredict(x);
        }
    }


    /**
    * DecisionTreeNode code
    */

    /**
     * Setter method for prediction variable
     * @param pred: Value to set prediction
     */
    void DecisionTreeNode::setPrediction(int pred){
        prediction = pred;
    }

    /**
     * Getter method for prediction variable
     * @return Value of prediction
     */
    int DecisionTreeNode::getPrediction(){
        return prediction;
    }

    /**
    * Setter method for parent node
    * @param parentNode: The parent of the current node
    */
    void DecisionTreeNode::setParent(DecisionTreeNode *parentNode) {
        parent = parentNode;
    }


    /**
    * Setter method for the value of the query
    * @param val
    */
    void DecisionTreeNode::setValue(double val) {
        value = val;

    }

    /**
     * Setter method for the query column
     * @param col
     */
    void DecisionTreeNode::setColumn(int col) {
        column = col;
    }


    /**
     * Get the left child
     * @return Left child node
     */
    DecisionTreeNode *DecisionTreeNode::getLeftChild() {
        return leftChild;
    }

    /**
     * Get the right child node
     * @return Right child node
     */
    DecisionTreeNode *DecisionTreeNode::getRightChild() {
        return rightChild;
    }

    /**
     * Create a new left child node
     */
    void DecisionTreeNode::createLeftChild() {

        leftChild = new DecisionTreeNode();
        leftChild->setParent(this);
    }

    /**
     * Create a new right child node
     */
    void DecisionTreeNode::createRightChild() {
        rightChild = new DecisionTreeNode();
        rightChild->setParent(this);
    }

    /**
     * Create left child node if possible, else create right child node
     */
    void DecisionTreeNode::createChild() {
        if (leftChild == nullptr) {
            createLeftChild();
        } else if (rightChild == nullptr) {
            createRightChild();
        }
    }

    /**
     * Calculate the Gini impurity.
     * https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
     *
     * @param X: Feature vectors
     * @param Y: Target classes
     * @return Calculated Gini impurity
     */
    double DecisionTreeNode::giniImpurity(std::vector<std::vector<double>> X, std::vector<double> Y) {
        double impurity = 1;
        int uniqueCount = (int) std::set<int>(Y.begin(), Y.end()).size();

        std::vector<int> classCount(uniqueCount, 0);

        for (double y : Y) {
            classCount[y]++;
        }

        for (int nClass : classCount) {
            impurity -= std::pow(nClass / (double) Y.size(), 2);
        }
        return impurity;
    }

    /**
     * Respond to a query.
     * @param x: Input feature vector
     * @return True if the feature is greater than or equal to the learned value else False
     */
    short int DecisionTreeNode::query(std::vector<double> x) const {
        return x[column] >= value;
    }


    /**
     * Partition into true and false
     * @param X: Feature vectors
     * @param Y: Target classes
     * @return
     */
    std::tuple<std::vector<std::vector<double>>,
            std::vector<double>,
            std::vector<std::vector<double>>,
            std::vector<double>> DecisionTreeNode::partition(const std::vector<std::vector<double>> &X,
                                                             const std::vector<double> &Y) {

        std::vector<std::vector<double>> part_trueX, part_falseX;
        std::vector<double> part_trueY, part_falseY;

        for (size_t i = 0; i < X.size(); i++) {
            if (query(X[i])) {
                part_trueX.push_back(X[i]);
                part_trueY.push_back(Y[i]);
            } else {
                part_falseX.push_back(X[i]);
                part_falseY.push_back(Y[i]);
            }
        }
        return std::make_tuple(part_trueX, part_trueY, part_falseX, part_falseY);
    }

    /**
     * Calculate information to be gained from the split.
     * https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
     *
     * @param part_trueX: Partition whereby x=true
     * @param part_trueY: Partition whereby y=true
     * @param part_falseX: Partition whereby x=false
     * @param part_falseY: Partition whereby y=false
     * @param uncertainty: The current uncertainty calculated using gini impurity
     *
     * @return Quantification of information to be gained for this split.
     */
    double DecisionTreeNode::informationGain(std::vector<std::vector<double>> part_trueX,
                                             std::vector<double> part_trueY,
                                             std::vector<std::vector<double>> part_falseX,
                                             std::vector<double> part_falseY,
                                             double uncertainty) {

        double p = (double) part_trueX.size() / (double) part_trueX.size() + (double) part_falseX.size();
        return uncertainty - p * giniImpurity(part_trueX, part_trueY) -
               (1 - p) * giniImpurity(part_falseX, part_falseY);
    }

    /**
     * Find the split that maximises information gain.
     *
     * 1. Reformat X to column-major set for contiguous memory access
     * 2. Iterate over rows and columns, trying all combos of column and potential value
     * 3. Partition the data `t each iteration and calculate information gain
     * 4. bestGain = gain if gain is greater than bestGain
     *
     * @param X: Feature vectors
     * @param Y: Target classes
     * @return Tuple like (gain, query)
     */
    std::tuple<double, std::pair<int, double>>
    DecisionTreeNode::findBestSplit(std::vector<std::vector<double>> X, std::vector<double> Y) {
        double bestGain, gain = 0;
        std::tuple<int, double> bestQuery; // int, double pair to store best query
        double uncertainty = giniImpurity(X, Y);
        int nFeatures = X[0].size();
        int nRows = X.size();
        std::vector<std::set<double>> uniqueItems(nFeatures);

        std::tuple<std::vector<std::vector<double>>,
                std::vector<double>,
                std::vector<std::vector<double>>,
                std::vector<double>> partitionedXY;

        // Put unique values in column-major sets.
        // This is precomputed here to provide contiguous memory access in the following code.
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nFeatures; j++) {
                uniqueItems[j].emplace(X[i][j]);
            }
        }

        for (int k = 0; k < nFeatures; k++) {
            for (double featureValue : uniqueItems[k]) {
                setColumn(k);
                setValue(featureValue);
                partitionedXY = partition(X, Y);

                if (std::get<0>(partitionedXY).empty() || std::get<3>(partitionedXY).empty()) {
                    continue;
                }

                gain = informationGain(std::get<0>(partitionedXY),
                                       std::get<1>(partitionedXY),
                                       std::get<2>(partitionedXY),
                                       std::get<3>(partitionedXY),
                                       uncertainty);

                if (gain > bestGain) {
                    bestGain = gain;
                    bestQuery = std::pair<int, double>(k, featureValue);
                }
            }
        }
        return {bestGain, bestQuery};
    }
}

