//
// Created by George Lancaster on 08/05/2020.
//

#include "FileReader.h"
#include "../libs/csv.h"

Matrix ml4cpp::FileReader::readSimpleCsv(std::string file) {

    io::CSVReader<2> in(file);
    double x; double y;
    Matrix ret(2);

    in.read_header(io::ignore_extra_column, "X", "Y");
    while(in.read_row(x, y)){
        ret[0].push_back(x);
        ret[1].push_back(y);
    }
    return ret;
}

Matrix ml4cpp::FileReader::readMultipleCsv(std::string file) {
    io::CSVReader<5> in(file);
    double a, b, c, d, y;
    std::vector<double> cool;

    Matrix ret(5);

    in.read_header(io::ignore_extra_column, "A", "B", "C", "D", "Y");

    while(in.read_row(a, b, c, d, y)){
        ret[0].push_back(a);
        ret[1].push_back(b);
        ret[2].push_back(c);
        ret[3].push_back(d);
        ret[4].push_back(y);

    }
    return ret;

}
