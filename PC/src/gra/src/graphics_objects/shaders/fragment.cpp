#include "logging.h"

#include "gra/graphics_objects/shaders/fragment.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			Fragment::Fragment(void)
				: Base(GL_FRAGMENT_SHADER)
			{
				TRACE("SHADER: FRAGMENT: CONSTRUCTED: {0}", (void*) this);
			}
			Fragment::Fragment(const std::string& filepath_or_source)
				: Base(GL_FRAGMENT_SHADER, filepath_or_source)
			{
				TRACE("SHADER: FRAGMENT: CONSTRUCTED: {0}", (void*) this);
			}

			Fragment::~Fragment(void)
			{
				TRACE("SHADER: FRAGMENT: DESTRUCTED: {0}", (void*) this);
			}

			void Fragment::copy(const Fragment& other)
			{
				Shader::Base::copy(other);
			}
			void Fragment::move(Fragment&& other)
			{
				Shader::Base::move(std::move(other));
			}
		}
	}
}
