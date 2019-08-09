#ifndef _GRA_SHADER_H_
#define _GRA_SHADER_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <unordered_map>
#include <inttypes.h>

#include "gra/gldebug.h"

namespace Gra
{
	class Shader
	{
		private:
			unsigned int m_renderer_id;
			std::unordered_map<std::string, int> m_uniform_location_chache;

		public:
			Shader(void);
			Shader(const std::string& path);
			~Shader(void);

		public:
			int get_uniform_location(const std::string& name);
			void set_uniform_4f(const std::string& name, float v0, float v1, float v2, float v3);
			void set_uniform_1i(const std::string& name, int v0);
			void set_uniform_mat4f(const std::string& name, glm::mat4 m0);

			void bind(void)   const;
			void unbind(void) const;
	};

	inline void Shader::bind(void) const
	{
		glCall(glUseProgram(m_renderer_id));
	}

	inline void Shader::unbind(void) const
	{
		glCall(glUseProgram(0));
	}
}

#endif
