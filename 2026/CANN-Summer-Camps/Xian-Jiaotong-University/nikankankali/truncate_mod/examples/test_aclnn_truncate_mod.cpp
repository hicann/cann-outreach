#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// TruncateMod golden: y = x1 - trunc(x1/x2) * x2
template<typename T>
std::vector<T> golden_truncate_mod(const std::vector<T>& x1, const std::vector<T>& x2) {
    std::vector<T> y(x1.size());
    for (size_t i = 0; i < x1.size(); ++i) {
        float q = std::trunc(static_cast<float>(x1[i]) / static_cast<float>(x2[i]));
        y[i] = static_cast<T>(static_cast<float>(x1[i]) - q * static_cast<float>(x2[i]));
    }
    return y;
}

int main() {
    std::cout << "TruncateMod example" << std::endl;
    std::vector<float> x1 = {7.0f, -7.0f, 7.0f, -7.0f};
    std::vector<float> x2 = {3.0f, 3.0f, -3.0f, -3.0f};
    auto y = golden_truncate_mod(x1, x2);
    std::cout << "x1: ";
    for (auto v: x1) std::cout << v << " ";
    std::cout << "\nx2: ";
    for (auto v: x2) std::cout << v << " ";
    std::cout << "\ny:  ";
    for (auto v: y) std::cout << v << " ";
    // Expected: 7%3=1, -7%3=-1, 7%-3=1, -7%-3=-1
    std::cout << "\nExpected: 1 -1 1 -1" << std::endl;
    return 0;
}
