#ifndef _GRA_SHADER_H_
#define _GRA_SHADER_H_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <memory>
#include <unordered_map>
#include <inttypes.h>

#include "gra/glstd.h"
#include "gra/gldebug.h"

namespace Gra
{
	class Shader
	{
		private:
			unsigned int m_renderer_id;
			std::unordered_map<std::string, int> m_uniform_location_cache;

		public:
			Shader(void);
			Shader(const std::string& path);
			Shader(const std::string& vertex_shader_source, const std::string& fragment_shader_source);
			~Shader(void);

		public:
			int get_uniform_location(const std::string& dirpath);

			void set_uniform(const std::string& name, float val0);
			void set_uniform(const std::string& name, float val0, float val1, float val2, float val3);
			void set_uniform(const std::string& name, int val0);
			void set_uniform(const std::string& name, glm::vec4 vec0);
			void set_uniform(const std::string& name, glm::mat4 mat0);

			void bind(void)   const;
			void unbind(void) const;
		public:
			static Shader& load(const std::string& dirpath);
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
