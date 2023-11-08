#include "misc.h"
#include <iostream>

void debug(std::string func_name, int line, std::string message)
{
    std::cout << "[Debug] function: '" << func_name << "' line: '" << line << "' message: '" << message << "'\n";
}
