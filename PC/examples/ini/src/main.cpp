#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <algorithm>
#include <signal.h>
#include <stdlib.h>

#include <ini/ini.h>

#include "commands.h"

static void finish(int sig)
{
	(void) sig;

	exit(0);
}

int main(void)
{
	signal(SIGINT, finish);

	while (1) {
		std::string       line;
		std::stringstream line_stream;
		std::string       statement;

		std::cout << examined_section << prompt;

		getline(std::cin, line);
		line_stream << line << ';';

		while (getline(line_stream, statement, ';')) {
			std::string              command;
			std::vector<std::string> args;

			statement.erase(0, statement.find_first_not_of(' '));

			if (statement.find_first_of(' ') == std::string::npos) {
				command = statement;
			} else {
				command = statement.substr(0, statement.find_first_of(' '));

				size_t first_arg_pos = statement.find_first_not_of(' ', command.size());
				std::string args_string = statement.substr(first_arg_pos);

				if (std::count(args_string.begin(), args_string.end(), '"') % 2 != 0) {
					std::cout << "[-] EXPECTED ‘\"’" << std::endl;
					exit(1);
				}

				char stopper = args_string[0] == '"' ? '"' : ' ';

				args_string.push_back(' ');
				for (std::string::iterator it = args_string.begin() + (stopper == '"' ? 1 : 0); it < args_string.end(); it++) {
					static std::string arg = "";

					if (*it == stopper) {
						it++;
						while (*it == ' ')
							it++;

						if (*it == '"')
							stopper = '"';
						else {
							stopper = ' ';
							it--;
						}

						args.push_back(arg);
						arg.clear();
					} else {
						arg.push_back(*it);
					}
				}
			}

			if (commands.find(command) != commands.end())
				commands.find(command)->second(args);
			else
				std::cout << "[-] COMMAND NOT FOUND" << std::endl;
		}
	}

	return 0;
}
