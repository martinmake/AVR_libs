#ifndef _GRA_GRAPHICS_OBJECT_SHADER_H_
#define _GRA_GRAPHICS_OBJECT_SHADER_H_

#include <string>

#include "gra/core.h"
#include "gra/graphics_objects/shaders/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			extern GLenum         filepath_to_type(const std::string& filepath);
			extern Shader::Base&& load            (const std::string& filepath);
		}
	}
}

#endif
