/*
 * AbstractRegressorModel.cpp
 *
 *  Created on: 22 Mar 2020
 *      Author: george
 */

#include "AbstractRegressorModel.h"

namespace ml4cpp {

    template <class T>
    AbstractRegressorModel<T>::AbstractRegressorModel() {
        // TODO Auto-generated constructor stub

    }

    template <class T>
    AbstractRegressorModel<T>::~AbstractRegressorModel() {
        // TODO Auto-generated destructor stub
    }

    template <class T>
    double AbstractRegressorModel<T>::predict(const T &x) {
        return 0;
    }

    template <class T>
    void AbstractRegressorModel<T>::fit(std::vector<T> X, std::vector<T> Y) {

    }

    template <class T>
    void AbstractRegressorModel<T>::setCoefficients(std::vector<T> coeff) {
        coefficients = coeff;
    }

} /* namespace ml4cpp */
