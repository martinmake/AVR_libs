#include "primitives/base.h"

namespace Gpl
{
	namespace Primitive
	{
		Base::Base(const glm::vec4& initial_color)
			: m_color(initial_color)
		{
		}
		Base::~Base(void)
		{
		}
	}
}
