#pragma once

#include <vector>
namespace CommonUtils {
    template <typename T>
    constexpr T sq(T x) {
        return x * x;
    }
}

std::vector<double> readVector(const std::string& filename);

std::vector<std::vector<double>> readMatrix(const std::string& filename);

std::pair<double, double> mean_err(std::vector<double>& vec);