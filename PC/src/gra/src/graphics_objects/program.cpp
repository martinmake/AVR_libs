#include <unordered_map>

#include <boost/filesystem.hpp>

#include "logging.h"

#include "gra/graphics_objects/program.h"
#include "gra/graphics_objects/shader.h"

namespace Gra
{
	namespace GraphicsObject
	{
		Program::Program(void)
		{
			glCall(m_renderer_id = glCreateProgram());
			TRACE("GL: PROGRAM: CREATED: {0}", m_renderer_id);
			TRACE("PROGRAM: CONSTRUCTED: {0}", (void*) this);
		}
		Program::Program(const std::string& dirpath)
			: Program()
		{
			load_shaders(dirpath);
		}
		Program::Program(const std::vector<Shader::Base>& shaders)
			: Program()
		{
			link_shaders(shaders);
		}

		Program::~Program(void)
		{
			if (m_renderer_id)
			{
				glCall(glDeleteProgram(m_renderer_id));
				TRACE("GL: PROGRAM: DELETED: {0}", m_renderer_id);
			}
			TRACE("PROGRAM: DESTRUCTED: {0}", (void*) this);
		}

		int Program::get_uniform_location(const std::string& name)
		{
			static std::unordered_map<std::string, unsigned int> uniform_location_cache;

			if (uniform_location_cache.find(name) != uniform_location_cache.end())
				return uniform_location_cache[name];

			int location;
			bind();
			glCall(location = glGetUniformLocation(m_renderer_id, name.c_str()));

			uniform_location_cache[name] = location;

			if (location == -1)
				std::cerr << "[INVALID UNIFORM] " << name << std::endl;

			return location;
		}

		void Program::set_uniform(const std::string& name, float val0)
		{
			int location = get_uniform_location(name);

			if (location == -1)
				return;

			bind();
			glCall(glUniform1f(location, val0));
		}
		void Program::set_uniform(const std::string& name, float val0, float val1, float val2, float val3)
		{
			int location = get_uniform_location(name);

			if (location == -1)
				return;

			bind();
			glCall(glUniform4f(location, val0, val1, val2, val3));
		}
		void Program::set_uniform(const std::string& name, int val0)
		{
			int location = get_uniform_location(name);

			if (location == -1)
				return;

			bind();
			glCall(glUniform1i(location, val0));
		}
		void Program::set_uniform(const std::string& name, Math::vec4<float> vec0)
		{
			int location = get_uniform_location(name);

			if (location == -1)
				return;

			bind();
			glCall(glUniform4fv(location, 1, &vec0.x));
		}
		void Program::set_uniform(const std::string& name, glm::vec4 vec0)
		{
			int location = get_uniform_location(name);

			if (location == -1)
				return;

			bind();
			glCall(glUniform4fv(location, 1, &vec0[0]));
		}
		void Program::set_uniform(const std::string& name, glm::mat4 mat0)
		{
			int location = get_uniform_location(name);

			if (location == -1)
				return;

			bind();
			glCall(glUniformMatrix4fv(location, 1, GL_FALSE, &mat0[0][0]));
		}

		void Program::load_shaders(const std::string& dirpath)
		{
			std::vector<Shader::Base> shaders;

			namespace fs = boost::filesystem;
			for (const fs::directory_entry& entry : fs::directory_iterator(dirpath))
				shaders.emplace_back(Shader::load(entry.path().string()));

			link_shaders(shaders);
		}

		void Program::link_shaders(const std::vector<Shader::Base>& shaders)
		{
			for (const Shader::Base& shader : shaders)
				glCall(glAttachShader(m_renderer_id, shader.renderer_id()));

			glCall(glLinkProgram    (m_renderer_id));
			glCall(glValidateProgram(m_renderer_id));
		}
	}
}
