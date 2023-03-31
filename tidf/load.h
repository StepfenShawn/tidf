#ifndef _LOAD_H_
#define _LOAD_H_

#include <fstream>
#include <regex>

// Load matrix from .txt file
template <class T>
Matrix<T> load_mat(std::string url) {
    std::vector<T> temp_line;
    std::vector<std::vector<T> > mat_arr;
    std::string line;
    std::ifstream in(url);
    if (!in.good())
        throw std::invalid_argument("Cannot open the file: " + url);
    std::regex pat_regex("[[:digit:]]+");

    while (std::getline(in, line)) {
        for (std::sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; 
                    it != end_it; ++it) {
            if (std::is_same<T, double>::value)
                temp_line.push_back(std::stod(it->str()));
            else if (std::is_same<T, int>::value)
                temp_line.push_back(std::stoi(it->str()));
            else if (std::is_same<T, float>::value)
                temp_line.push_back(std::stof(it->str()));
        }
        mat_arr.push_back(temp_line);
        temp_line.clear();
    }

    return Matrix<T>(mat_arr);
}

#endif /* _LOAD_H_  */