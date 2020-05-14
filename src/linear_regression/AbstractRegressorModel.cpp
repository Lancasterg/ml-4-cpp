/*
 * AbstractRegressorModel.cpp
 *
 *  Created on: 22 Mar 2020
 *      Author: george
 */

#include "AbstractRegressorModel.h"

namespace ml4cpp {

    AbstractRegressorModel::AbstractRegressorModel() {
        // TODO Auto-generated constructor stub

    }

    AbstractRegressorModel::~AbstractRegressorModel() {
        // TODO Auto-generated destructor stub
    }

    double AbstractRegressorModel::predict(const double &x) {
        return 0;
    }

    void AbstractRegressorModel::fit(std::vector<double> X, std::vector<double> Y) {

    }

    void AbstractRegressorModel::setCoefficients(std::vector<double> coeff) {
        coefficients = coeff;
    }

} /* namespace ml4cpp */
