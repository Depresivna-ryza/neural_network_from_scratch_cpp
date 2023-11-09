#ifndef MISC_H
#define MISC_H

#include <iostream>
#include <string>

#define DEBUG(message) debug(__FUNCTION__, __LINE__, message)

void debug(std::string func_name, int line, std::string message) {
    std::cout << "[Debug] function: '" << func_name << "' line: '" << line << "' message: '"
              << message << "'\n";
}

#endif // MISC_H
