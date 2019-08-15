#ifndef _GRA_GRAPHICS_OBJECT_SHADER_H_
#define _GRA_GRAPHICS_OBJECT_SHADER_H_

#include <string>

#include "gra/glstd.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			extern GLenum filepath_to_type(const std::string& filepath);
		}
	}
}

#endif
