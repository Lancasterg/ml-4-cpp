#include "../libs/catch.hpp"
#include "../decision_tree/DecisionTreeClassifier.h"


TEST_CASE("Test Gini impurity ", "[classic]") {
    ml4cpp::DecisionTreeNode treeNode;

    std::vector<std::vector<double>> X = {{1, 1},
                                          {1, 1},
                                          {1, 1},
                                          {1, 1}};
    std::vector<double> Y = {0, 1, 1, 0};

    REQUIRE(treeNode.giniImpurity(X, Y) == 0.5);

    X = {{1},
         {1},
         {1},
         {1}};
    Y = {0, 0, 0, 0};

    REQUIRE(treeNode.giniImpurity(X, Y) == 0);

    X = {{1},
         {1},
         {1},
         {0}};
    Y = {0, 0, 0, 1};

    REQUIRE(treeNode.giniImpurity(X, Y) == 0.375);


}


TEST_CASE("Test DecisionTreeNode ", "[classic]") {

    ml4cpp::DecisionTreeNode treeNode;
    treeNode.createChild();
    treeNode.createChild();
    treeNode.getLeftChild();
    treeNode.getRightChild();


}