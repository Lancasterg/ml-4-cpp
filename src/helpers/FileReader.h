//
// Created by George Lancaster on 08/05/2020.
//

#ifndef ML_4_CPP_FILEREADER_H
#define ML_4_CPP_FILEREADER_H

#include <string>
#include <vector>
#include "../linear_algebra/linalg.h"
#include "../libs/csv.h"


namespace ml4cpp {

    class FileReader {
    public:
        Matrix readSimpleCsv(std::string file);
        Matrix readMultipleCsv(std::string file);


    };
}

#endif //ML_4_CPP_FILEREADER_H
