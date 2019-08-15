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
			}
			Fragment::Fragment(const std::string& filepath_or_source)
				: Base(GL_FRAGMENT_SHADER, filepath_or_source)
			{
			}

			Fragment::~Fragment(void)
			{
			}

			void Fragment::copy(const Fragment& other)
			{
				Shader::Fragment::copy(other);
			}
			void Fragment::move(Fragment&& other)
			{
				Shader::Fragment::move(std::move(other));
			}
		}
	}
}
