#include <iostream>
#include <fstream>
#include <sstream>

#include "shader.h"

static unsigned int compile_shader(unsigned int type, const std::string& source, const std::string& key = "");
static unsigned int create_shader(const std::string& dirpath);
static unsigned int link_shader(const unsigned int vertex_shader_object, unsigned int fragment_shader_object);

namespace Gra
{
	Shader::Shader(void)
	{
	}
	Shader::Shader(const std::string& dirpath)
	{
		m_renderer_id = create_shader(dirpath);
		bind();
	}
	Shader::Shader(const std::string& vertex_shader_source, const std::string& fragment_shader_source)
	{
		unsigned int vertex_shader_object   = compile_shader(GL_VERTEX_SHADER,     vertex_shader_source);
		unsigned int fragment_shader_object = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
		m_renderer_id = link_shader(vertex_shader_object, fragment_shader_object);
		bind();
	}

	Shader::~Shader(void)
	{
		glCall(glDeleteProgram(m_renderer_id));
	}

	int Shader::get_uniform_location(const std::string& name)
	{
		bind();
		if (m_uniform_location_cache.find(name) != m_uniform_location_cache.end())
			return m_uniform_location_cache[name];

		int location;
		glCall(location = glGetUniformLocation(m_renderer_id, name.c_str()));

		m_uniform_location_cache[name] = location;

		if (location == -1)
			std::cerr << "[INVALID UNIFORM] " << name << std::endl;

		return location;
	}

	void Shader::set_uniform(const std::string& name, float val0)
	{
		bind();
		int location = get_uniform_location(name);

		if (location == -1)
			return;

		glCall(glUniform1f(location, val0));
	}
	void Shader::set_uniform(const std::string& name, float val0, float val1, float val2, float val3)
	{
		bind();
		int location = get_uniform_location(name);

		if (location == -1)
			return;

		glCall(glUniform4f(location, val0, val1, val2, val3));
	}
	void Shader::set_uniform(const std::string& name, int val0)
	{
		bind();
		int location = get_uniform_location(name);

		if (location == -1)
			return;

		glCall(glUniform1i(location, val0));
	}
	void Shader::set_uniform(const std::string& name, glm::vec4 vec0)
	{
		bind();
		int location = get_uniform_location(name);

		if (location == -1)
			return;

		glCall(glUniform4fv(location, 1, &vec0[0]));
	}
	void Shader::set_uniform(const std::string& name, glm::mat4 mat0)
	{
		bind();
		int location = get_uniform_location(name);

		if (location == -1)
			return;

		glCall(glUniformMatrix4fv(location, 1, GL_FALSE, &mat0[0][0]));
	}
}

static std::string load_shader(const std::string& filepath)
{
	std::ifstream stream(filepath);
	if (!stream.is_open())
		std::cout << "[SHADER] SHADER NOT FOUND '" << filepath << '`' << std::endl;
	assert(stream.is_open());
	std::stringstream source_buffer;
	source_buffer << stream.rdbuf();

	return source_buffer.str();
}

static unsigned int compile_shader(unsigned int type, const std::string& source, const std::string& key)
{
	static std::unordered_map<std::string, unsigned int> shader_object_cache;
	unsigned int id;

	if (!key.empty() && shader_object_cache.find(key) != shader_object_cache.end())
		id = shader_object_cache[key];
	else
	{
		glCall(id = glCreateShader(type));

		const char* src = source.c_str();
		glCall(glShaderSource(id, 1, &src, nullptr));
		glCall(glCompileShader(id));

		int result;
		glCall(glGetShaderiv(id, GL_COMPILE_STATUS, &result));
		if (result == GL_FALSE) {
			int length;
			glCall(glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length));
			char* message = (char *) alloca(length * sizeof(char));
			glCall(glGetShaderInfoLog(id, length, &length, message));
			std::cout << "Failed to compile " << (type == GL_VERTEX_SHADER ?  "VERTEX" : "FRAGMENT") << " shader!" << std::endl;
			std::cout << message << std::endl;
			glCall(glDeleteShader(id));
			return 0;
		}

		shader_object_cache[key] = id;
	}
	return id;
}

static unsigned int create_shader(const std::string& dirpath)
{
	unsigned int vs = compile_shader(GL_VERTEX_SHADER,   load_shader(dirpath + "/" + "vertex.vs"  ), dirpath + "v");
	unsigned int fs = compile_shader(GL_FRAGMENT_SHADER, load_shader(dirpath + "/" + "fragment.fs"), dirpath + "f");

	return link_shader(vs, fs);
}

static unsigned int link_shader(const unsigned int vertex_shader_object, unsigned int fragment_shader_object)
{
	unsigned int program;

	glCall(program = glCreateProgram());

	glCall(glAttachShader(program,   vertex_shader_object));
	glCall(glAttachShader(program, fragment_shader_object));

	glCall(glLinkProgram(program));
	glCall(glValidateProgram(program));

	return program;
}
