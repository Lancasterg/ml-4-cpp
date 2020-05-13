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
    private:
        std::vector<std::vector<double>> readMultipleCsvCm_4(std::string file);
        std::vector<std::vector<double>> readMultipleCsvCm_5(std::string file);

        std::vector<std::vector<double>> readSimpleCsvRm(std::string basicString);
        std::vector<std::vector<double>> readMultipleCsvRm_3(std::string file);
        std::vector<std::vector<double>> readMultipleCsvRm_4(std::string file);
        std::vector<std::vector<double>> readMultipleCsvRm_5(std::string file);

    public:
        FileReader() = default;

        Matrix readSimpleCsvCm(std::string file);
        static std::vector<double> getRow(int rowNum, Matrix data);


        std::vector<std::vector<double>> readCsvCm(std::string file, int numCols);
        std::vector<std::vector<double>> readCsvRm(std::string file, int numCols);



    };
}

#endif //ML_4_CPP_FILEREADER_H
