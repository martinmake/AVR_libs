#ifndef _INIPARSER_COMMANDS_H_
#define _INIPARSER_COMMANDS_H_

#include <string>
#include <map>

#include <ini/ini.h>

using command_func = void (*)(const std::vector<std::string>& args);

extern std::map<const std::string, command_func> commands;
extern Ini ini;
extern std::string examined_section;
extern std::string prompt;

extern void list(const std::vector<std::string>& args);
extern void examine(const std::vector<std::string>& args);
extern void exit(const std::vector<std::string>& args);

#endif
