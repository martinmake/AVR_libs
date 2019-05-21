#ifndef _INIPARSER_INI_H_
#define _INIPARSER_INI_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <iniparser.h>
#include <alloca.h>

class Ini
{
	private:
		dictionary* m_ini;

	public:
		Ini(const std::string& path);
		Ini();
		~Ini();

		int                      get_section_count    (void);
		std::string              get_section_name     (int n);
		int                      get_section_key_count(const std::string& section);
		std::vector<std::string> get_section_keys     (const std::string& section);
		std::string              get                  (const std::string& section, const std::string& key);
		bool                     find_section         (const std::string& section);
		void                     add_section          (const std::string& section);
		void                     remove_section       (const std::string& section);
		void                     set                  (const std::string& section, const std::string& key, const std::string& val);
		void                     unset                (const std::string& section, const std::string& key);
		void                     dump                 (const std::string& path);
		void                     dump                 (FILE* file);
		void                     load                 (const std::string& file);
		void                     free_dict            (void);
};

inline int Ini::get_section_count(void)
{
	return iniparser_getnsec(m_ini);
}

inline std::string Ini::get_section_name(int n)
{
	return iniparser_getsecname(m_ini, n);
}

inline int Ini::get_section_key_count(const std::string& section)
{
	return iniparser_getsecnkeys(m_ini, section.c_str());
}

inline std::vector<std::string> Ini::get_section_keys(const std::string& section)
{
	int key_count = get_section_key_count(section);
	const char** keys = (const char**) alloca(key_count);

	iniparser_getseckeys(m_ini, section.c_str(), keys);

	std::vector<std::string> key_vect(keys, keys + key_count);
	for (std::string& key : key_vect)
		key.erase(0, key.find_first_of(':') + 1);

	return key_vect;
}

inline std::string Ini::get(const std::string& section, const std::string& key)
{
	std::string ini_key = section + ':' + key;
	return iniparser_getstring(m_ini, ini_key.c_str(), "");
}

inline bool Ini::find_section(const std::string& section)
{
	return iniparser_find_entry(m_ini, section.c_str());
}

inline void Ini::add_section(const std::string& section)
{
	iniparser_set(m_ini, section.c_str(), NULL);
}

inline void Ini::remove_section(const std::string& section)
{
	iniparser_unset(m_ini, section.c_str());
}

inline void Ini::set(const std::string& section, const std::string& key, const std::string& val)
{
	const std::string entry = section + ':' + key;

	if (!find_section(section))
		add_section(section);

	iniparser_set(m_ini, entry.c_str(), val.c_str());
}

inline void Ini::unset(const std::string& section, const std::string& key)
{
	const std::string entry = section + ':' + key;

	iniparser_unset(m_ini, entry.c_str());
}

inline void Ini::dump(const std::string& path)
{
	FILE* ini_file_pointer = fopen(path.c_str(), "w");

	iniparser_dump_ini(m_ini, ini_file_pointer);

	for (uint8_t i = 3; i && fclose(ini_file_pointer) == EOF; i--)
		std::cerr << "[-] UNABLE TO CLOSE INI FILE! (" << path << ')' << std::endl;
}

inline void Ini::dump(FILE* file)
{
	iniparser_dump_ini(m_ini, file);
}

inline void Ini::load(const std::string& path)
{
	free_dict();

	m_ini = iniparser_load(path.c_str());
}

inline void Ini::free_dict()
{
	if (m_ini == nullptr)
		return;

	iniparser_freedict(m_ini);
	m_ini = nullptr;
}

#endif
