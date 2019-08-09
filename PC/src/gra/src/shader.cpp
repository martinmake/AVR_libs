#include <iostream>
#include <fstream>
#include <sstream>

#include "shader.h"

struct ShaderSource
{
	std::string vertex,
		    fragment;
};

extern std::string load_shader(const std::string& filepath);
extern unsigned int create_shader(const std::string& filepath);
extern unsigned int compile_shader(unsigned int type, const std::string& source);

namespace Gra
{
	Shader::Shader(void)
	{
	}
	Shader::Shader(const std::string& filepath)
	{
		m_renderer_id = create_shader(filepath);
		bind();
	}

	Shader::~Shader(void)
	{
		glCall(glDeleteProgram(m_renderer_id));
	}

	int Shader::get_uniform_location(const std::string& name)
	{
		if (m_uniform_location_chache.find(name) != m_uniform_location_chache.end())
			return m_uniform_location_chache[name];

		int location;
		glCall(location = glGetUniformLocation(m_renderer_id, name.c_str()));

		m_uniform_location_chache[name] = location;

		if (location == -1)
			std::cerr << "[INVALID UNIFORM] " << name << std::endl;

		return location;
	}

	void Shader::set_uniform_4f(const std::string& name, float v0, float v1, float v2, float v3)
	{
		int location = get_uniform_location(name);

		if (location == -1)
			return;

		glCall(glUniform4f(location, v0, v1, v2, v3));
	}

	void Shader::set_uniform_1i(const std::string& name, int v0)
	{
		int location = get_uniform_location(name);

		if (location == -1)
			return;

		glCall(glUniform1i(location, v0));
	}

	void Shader::set_uniform_mat4f(const std::string& name, glm::mat4 m0)
	{
		int location = get_uniform_location(name);

		if (location == -1)
			return;

		glCall(glUniformMatrix4fv(location, 1, GL_FALSE, &m0[0][0]));
	}
}

std::string load_shader(const std::string& filepath)
{
	std::ifstream stream(filepath);
	if (!stream.is_open())
		std::cout << "[SHADER] SHADER NOT FOUND '" << filepath << '`' << std::endl;
	assert(stream.is_open());
	std::stringstream source_buffer;
	source_buffer << stream.rdbuf();

	return source_buffer.str();
}

unsigned int create_shader(const std::string& filepath)
{
	unsigned int program;
	unsigned int vs = compile_shader(GL_VERTEX_SHADER,   load_shader(filepath + "/vertex.vs"));
	unsigned int fs = compile_shader(GL_FRAGMENT_SHADER, load_shader(filepath + "/fragment.fs"));

	glCall(program = glCreateProgram());

	glCall(glAttachShader(program, vs));
	glCall(glAttachShader(program, fs));

	glCall(glLinkProgram(program));
	glCall(glValidateProgram(program));

	glCall(glDeleteShader(vs));
	glCall(glDeleteShader(fs));

	return program;
}

unsigned int compile_shader(unsigned int type, const std::string& source)
{
	unsigned int id;
	const char* src = source.c_str();
	int result;

	glCall(id = glCreateShader(type));

	glCall(glShaderSource(id, 1, &src, nullptr));
	glCall(glCompileShader(id));

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

	return id;
}
