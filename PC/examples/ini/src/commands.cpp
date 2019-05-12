#include <iostream>
#include <sstream>

#include "commands.h"

std::string examined_section = "";
std::string prompt = "> ";

Ini ini;

void load(const std::vector<std::string>& args)
{
	if (args.size() != 1) {
		std::cout << "[-] EXPECTED 1 ARGUMENT, GOT " << args.size() << std::endl;
		return;
	}

	examined_section = "";

	const std::string path = args[0];
	ini.load(path);
}

void dump(const std::vector<std::string>& args)
{
	if (args.size() > 1) {
		std::cout << "[-] EXPECTED 1 ARGUMENT, GOT " << args.size() << std::endl;
		return;
	}

	if (args.empty() || args[0] == "-")
		ini.dump(stdout);
	else
		ini.dump(args[0]);
}

void list(const std::vector<std::string>& args)
{
	if (!examined_section.empty()) {
		std::vector<std::string> keys = ini.get_section_keys(examined_section);
		printf("[%s]\n", examined_section.c_str());
		for (std::string key : keys)
			std::cout << key << " = " << ini.get_string(examined_section, key) << std::endl;
	} else if (args.empty()) {
		int section_count = ini.get_section_count();
		for (int i = 0; i < section_count; i++)
			printf("[%s]\n", ini.get_section_name(i).c_str());
	} else {
		for (std::string arg : args) {
			std::vector<std::string> keys = ini.get_section_keys(arg);
			printf("[%s]\n", arg.c_str());
			for (std::string key : keys)
				std::cout << key << " = " << ini.get_string(arg, key) << std::endl;
		}
	}
}

void examine(const std::vector<std::string>& args)
{
	if (args.size() != 1) {
		std::cout << "[-] EXPECTED 1 ARGUMENT, GOT " << args.size() << std::endl;
		return;
	}

	const std::string& section = args[0];

	if (!ini.find_section(section)) {
		std::cout << "[-] SECTION NOT PRESENT" << std::endl;
		return;
	}

	examined_section = section;
}

void un_examine(const std::vector<std::string>& args)
{
	(void) args;

	examined_section = "";
}

void add(const std::vector<std::string>& args)
{
	if (args.empty()) {
		std::cout << "[-] EXPECTED ATLEAST 1 ARGUMENT, GOT " << args.size() << std::endl;
		return;
	}

	for (std::string section : args)
		ini.add_section(section);
}

void rm(const std::vector<std::string>& args)
{
	if (args.empty()) {
		std::cout << "[-] EXPECTED ATLEAST 1 ARGUMENT, GOT " << args.size() << std::endl;
		return;
	}

	for (std::string section : args)
		ini.remove_section(section);
}

void set(const std::vector<std::string>& args)
{
	if (args.size() != 2) {
		std::cout << "[-] EXPECTED 2 ARGUMENT, GOT " << args.size() << std::endl;
		return;
	}

	if (examined_section.empty()) {
		std::cout << "[-] NO EXAMINED SECTION" << std::endl;
		return;
	}

	const std::string& key = args[0];
	const std::string& val = args[1];
	ini.set(examined_section, key, val);
}

void unset(const std::vector<std::string>& args)
{
	if (args.empty()) {
		std::cout << "[-] EXPECTED ATLEAST 1 ARGUMENT, GOT " << args.size() << std::endl;
		return;
	}

	if (examined_section.empty()) {
		std::cout << "[-] NO EXAMINED SECTION" << std::endl;
		return;
	}

	const std::string& key = args[0];
	ini.unset(examined_section, key);
}

void exit(const std::vector<std::string>& args)
{
	(void) args;

	exit(0);
}

std::map<const std::string, command_func> commands = {
	{ "load",     load       },
	{ "dump",     dump       },
	{ "ls",       list       },
	{ "examine",  examine    },
	{ "ex",       examine    },
	{ "uexamine", un_examine },
	{ "uex",      un_examine },
	{ "add",      add        },
	{ "rm",       rm         },
	{ "set",      set        },
	{ "unset",    unset      },
	{ "exit",     exit       }
};
