//
// Created by George Lancaster on 08/05/2020.
//

#include "FileReader.h"
#include "../libs/csv.h"

namespace ml4cpp {

    Matrix FileReader::readSimpleCsvCm(std::string file) {

        io::CSVReader<2> in(file);
        double x;
        double y;
        Matrix ret(2);

        in.read_header(io::ignore_extra_column, "X", "Y");
        while (in.read_row(x, y)) {
            ret[0].push_back(x);
            ret[1].push_back(y);
        }
        return ret;
    }

    Matrix FileReader::readCsvCm(std::string file, int numCols) {
        if (numCols == 2) {
            return ml4cpp::FileReader::readSimpleCsvCm(file);
        } else if (numCols == 4) {
            return ml4cpp::FileReader::readMultipleCsvCm_4(file);
        } else if (numCols == 5) {
            return ml4cpp::FileReader::readMultipleCsvCm_5(file);
        }
    }

    Matrix FileReader::readMultipleCsvCm_5(std::string file) {
        io::CSVReader<5> in(file);
        double a, b, c, d, y;

        Matrix ret(5);

        in.read_header(io::ignore_extra_column, "A", "B", "C", "D", "Y");

        while (in.read_row(a, b, c, d, y)) {
            ret[0].push_back(a);
            ret[1].push_back(b);
            ret[2].push_back(c);
            ret[3].push_back(d);
            ret[4].push_back(y);

        }
        return ret;
    }

    Matrix FileReader::readMultipleCsvCm_4(std::string file) {
        io::CSVReader<4> in(file);
        double a, b, c, y;

        Matrix ret(4);

        in.read_header(io::ignore_extra_column, "A", "B", "C", "Y");

        while (in.read_row(a, b, c, y)) {
            ret[0].push_back(a);
            ret[1].push_back(b);
            ret[2].push_back(c);
            ret[3].push_back(y);

        }
        return ret;
    }

    std::vector<double> FileReader::getRow(int rowNum, Matrix data) {
        std::vector<double> row(data.size());

        // Create feature vector
        for (size_t j = 0; j < data.size(); j++) {
            row[j] = data[j][rowNum];
        }
        return row;
    }

    std::vector<std::vector<double>> FileReader::readCsvRm(std::string file, int numCols) {
        if (numCols == 2) {
            return readSimpleCsvRm(file);
        } else if (numCols == 3) {
            return readMultipleCsvRm_3(file);
        } else if (numCols == 4) {
            return readMultipleCsvRm_4(file);
        } else if (numCols == 5) {
            return readMultipleCsvRm_5(file);
        }
    }

    std::vector<std::vector<double>> FileReader::readSimpleCsvRm(std::string file) {
        io::CSVReader<2> in(file);
        std::vector<double> row;
        double a, y;
        Matrix ret;

        in.read_header(io::ignore_extra_column, "A", "Y");
        while (in.read_row(a, y)) {
            ret.push_back({a, y});
        }
        return ret;
    }

    std::vector<std::vector<double>> FileReader::readMultipleCsvRm_3(std::string file) {
        io::CSVReader<3> in(file);
        std::vector<double> row;
        double a, b, y;
        Matrix ret;

        in.read_header(io::ignore_extra_column, "A", "B", "Y");
        while (in.read_row(a, b, y)) {
            ret.push_back({a, b, y});
        }
        return ret;
    }

    std::vector<std::vector<double>> FileReader::readMultipleCsvRm_4(std::string file) {
        io::CSVReader<4> in(file);
        std::vector<double> row;
        double a, b, c, y;
        Matrix ret;

        in.read_header(io::ignore_extra_column, "A", "B", "C", "Y");
        while (in.read_row(a, b, c, y)) {
            ret.push_back({a, b, c, y});
        }
        return ret;
    }


    std::vector<std::vector<double>> FileReader::readMultipleCsvRm_5(std::string file) {
        io::CSVReader<5> in(file);
        std::vector<double> row;
        double a, b, c, d, y;
        Matrix ret;

        in.read_header(io::ignore_extra_column, "A", "B", "C", "D", "Y");
        while (in.read_row(a, b, c, d, y)) {
            ret.push_back({a, b, c, d, y});
        }
        return ret;
    }

}







