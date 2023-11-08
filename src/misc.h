#ifndef MISC_H
#define MISC_H

#include <string>

#define DEBUG(message) debug(__FUNCTION__, __LINE__, message)
void debug(std::string func_name, int line, std::string message);



#endif // MISC_H
