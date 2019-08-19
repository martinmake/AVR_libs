#include "logging.h"

#include "gra/graphics_objects/shaders/vertex.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			Vertex::Vertex(void)
				: Base(GL_VERTEX_SHADER)
			{
				TRACE("SHADER: VERTEX: CONSTRUCTED: {0}", (void*) this);
			}
			Vertex::Vertex(const std::string& filepath_or_source)
				: Base(GL_VERTEX_SHADER, filepath_or_source)
			{
				TRACE("SHADER: VERTEX: CONSTRUCTED: {0}", (void*) this);
			}

			Vertex::~Vertex(void)
			{
				TRACE("SHADER: VERTEX: DESTRUCTED: {0}", (void*) this);
			}

			void Vertex::copy(const Vertex& other)
			{
				Shader::Base::copy(other);
			}
			void Vertex::move(Vertex&& other)
			{
				Shader::Base::move(std::move(other));
			}
		}
	}
}
