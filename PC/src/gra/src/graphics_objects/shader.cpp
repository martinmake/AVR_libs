#include "gra/graphics_objects/shader.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			GLenum filepath_to_type(const std::string& filepath)
			{
				std::string extension = filepath.substr(filepath.length() - 2, 2);

				if      (extension == "vs") return GL_VERTEX_SHADER;
				else if (extension == "fs") return GL_FRAGMENT_SHADER;
				else if (extension == "gs") return GL_GEOMETRY_SHADER;
				else                        return 0;
			}
		}
	}
}

