#include "logging.h"

#include "gra/graphics_objects/shader.h"
#include "gra/graphics_objects/shaders/all.h"

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

			Shader::Base&& load(const std::string& filepath)
			{
				switch (filepath_to_type(filepath))
				{
					case GL_VERTEX_SHADER:   return std::move(*new Shader::Vertex  (filepath));
					case GL_FRAGMENT_SHADER: return std::move(*new Shader::Fragment(filepath));
					case GL_GEOMETRY_SHADER: return std::move(*new Shader::Geometry(filepath));
					default: ERROR("SHADER: UNKNOWN SHADER TYPE: {0}", filepath); abort();
				}
			}
		}
	}
}

