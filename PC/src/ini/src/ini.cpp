#include "ini.h"

Ini::Ini(const std::string& path)
{
	load(path.c_str());
}

Ini::Ini()
	: m_ini(nullptr)
{
}

Ini::~Ini()
{
	if (m_ini != nullptr)
		free_dict();
}
