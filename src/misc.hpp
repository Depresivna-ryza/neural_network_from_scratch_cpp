#ifndef MISC_H
#define MISC_H

#include <iostream>
#include <random>
#include <string>

#define DEBUG(message) debug(__FUNCTION__, __LINE__, message)

void debug(std::string func_name, int line, auto message) {
    std::cout << "[Debug] function: '" << func_name << "' line: '" << line << "' message: '"
              << message << "'\n";
}

double random_gaussian() {
    std::normal_distribution<double> d(0, 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    return d(gen);
}

#endif // MISC_H
