#include <utility>

#include "gra/graphics_objects/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		Base::Base(void)
		{
		}

		Base::~Base(void)
		{
		}

		void Base::copy(const Base& other)
		{
			(void) other;
		}
		void Base::move(Base&& other)
		{
			m_renderer_id = std::exchange(other.m_renderer_id, 0);
		}
	}
}
