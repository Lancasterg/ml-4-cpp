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


TEST_CASE("Test DecisionTreeNode partition ", "[classic]") {

    std::vector<std::vector<double>> X = {{500, 0},
                                          {100, 0},
                                          {0,   0.5},
                                          {0,   50}};
    std::vector<double> Y = {1, 0, 0, 0};


    std::vector<std::vector<double>> true_partX = {{500, 0},
                                                   {100, 0}};
    std::vector<std::vector<double>> false_partX = {{0, 0.5},
                                                    {0, 50}};
    std::vector<double> true_partY = {1, 0};
    std::vector<double> false_partY = {0, 0};

    std::tuple<std::vector<std::vector<double>>,
            std::vector<double>,
            std::vector<std::vector<double>>,
            std::vector<double>> result;

    ml4cpp::DecisionTreeNode treeNode;
    treeNode.setColumn(0);
    treeNode.setValue(50);

    result = treeNode.partition(X, Y);

    REQUIRE(std::get<0>(result) == true_partX);
    REQUIRE(std::get<1>(result) == true_partY);
    REQUIRE(std::get<2>(result) == false_partX);
    REQUIRE(std::get<3>(result) == false_partY);

}


TEST_CASE("Test DecisionTreeNode FindBestSplit ", "[classic]") {
    std::vector<std::vector<double>> X = {{1, 2},
                                          {3, 4},
                                          {5, 6},
                                          {7, 8}};
    std::vector<double> Y = {1, 0, 0, 0};
    ml4cpp::DecisionTreeNode treeNode;

    treeNode.findBestSplit(X, Y);


}


TEST_CASE("Test DecisionTreeFit", "[classic]") {
    ml4cpp::DecisionTreeClassifier model;
    std::vector<std::vector<double>> Xtrain = {{1,   10},
                                               {1.1, 10},
                                               {500, 1000},
                                               {550, 1100}};
    std::vector<double> Ytrain = {1, 1, 0, 0};

    std::vector<std::vector<double>> Xtest = {{500, 1000},
                                              {550, 1100},
                                              {1,   10},
                                              {1.1, 10}};


    std::vector<double> Ytest = {0, 0, 1, 1};

    model.fit(Xtrain, Ytrain);

    for (size_t i = 0; i < Ytest.size(); i++) {
        REQUIRE(model.predict(Xtest[i]) == Ytest[i]);
    }


}

TEST_CASE("Test DecisionTreeNode ", "[classic]") {

    ml4cpp::DecisionTreeNode treeNode;
    treeNode.createChild();
    treeNode.createChild();
    treeNode.getLeftChild();
    treeNode.getRightChild();


}