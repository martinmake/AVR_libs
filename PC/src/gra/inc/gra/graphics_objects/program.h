#ifndef _GRA_GRAPHICS_OBJECT_PROGRAM_H_
#define _GRA_GRAPHICS_OBJECT_PROGRAM_H_

#include <string>
#include <vector>

#include "gra/math.h"
#include "gra/glstd.h"
#include "gra/gldebug.h"

#include "gra/graphics_objects/base.h"
#include "gra/graphics_objects/shaders/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		class Program : public GraphicsObject::Base
		{
			public: // CONSTRUCTORS
				Program(void);
				Program(const std::string& dirpath);
				Program(const std::vector<Shader::Base>& shaders);

			public: // FUNCTIONS
				int get_uniform_location(const std::string& dirpath);

				void set_uniform(const std::string& name, float val0);
				void set_uniform(const std::string& name, float val0, float val1, float val2, float val3);
				void set_uniform(const std::string& name, int val0);
				void set_uniform(const std::string& name, glm::vec4 vec0);
				void set_uniform(const std::string& name, Math::vec4<float> vec0);
				void set_uniform(const std::string& name, glm::mat4 mat0);

				void load_shaders(const std::string& dirpath);
				void link_shaders(const std::vector<Shader::Base>& shaders);

				void bind(void)   const override;
				void unbind(void) const override;

			DECLARATION_MANDATORY(Program)
		};

		// FUNCTIONS
		inline void Program::  bind(void) const { glCall(glUseProgram(m_renderer_id)); }
		inline void Program::unbind(void) const { glCall(glUseProgram(0            )); }

		DEFINITION_MANDATORY(Program, )
	}
}

#endif
