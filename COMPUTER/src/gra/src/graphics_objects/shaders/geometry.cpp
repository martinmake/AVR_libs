#include "logging.h"

#include "gra/graphics_objects/shaders/geometry.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			Geometry::Geometry(void)
				: Base(GL_GEOMETRY_SHADER)
			{
				TRACE("SHADER: GEOMETRY: CONSTRUCTED: {0}", (void*) this);
			}
			Geometry::Geometry(const std::string& filepath_or_source)
				: Base(GL_GEOMETRY_SHADER, filepath_or_source)
			{
				TRACE("SHADER: GEOMETRY: CONSTRUCTED: {0}", (void*) this);
			}

			Geometry::~Geometry(void)
			{
				TRACE("SHADER: GEOMETRY: DESTRUCTED: {0}", (void*) this);
			}

			void Geometry::copy(const Geometry& other)
			{
				Shader::Base::copy(other);
			}
			void Geometry::move(Geometry&& other)
			{
				Shader::Base::move(std::move(other));
			}
		}
	}
}
