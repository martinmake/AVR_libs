#include "ini.h"

Ini::Ini(const std::string& path)
{
	load(path.c_str());
}

Ini::Ini()
{
	load("/dev/null");
}

Ini::~Ini()
{
	free_dict();
}
