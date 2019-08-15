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
			}
			Vertex::Vertex(const std::string& filepath_or_source)
				: Base(GL_VERTEX_SHADER, filepath_or_source)
			{
			}

			Vertex::~Vertex(void)
			{
			}

			void Vertex::copy(const Vertex& other)
			{
				Shader::Vertex::copy(other);
			}
			void Vertex::move(Vertex&& other)
			{
				Shader::Vertex::move(std::move(other));
			}
		}
	}
}
