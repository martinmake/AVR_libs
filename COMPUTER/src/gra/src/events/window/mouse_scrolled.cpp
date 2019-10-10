#include "logging.h"

#include "gra/events/window/mouse_scrolled.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			MouseScrolled::MouseScrolled(float initial_xoffset, float initial_yoffset)
				: m_xoffset(initial_xoffset), m_yoffset(initial_yoffset)
			{
			}
			MouseScrolled::~MouseScrolled(void)
			{
			}

			void MouseScrolled::copy(const MouseScrolled& other)
			{
				Event::Window::Base::copy(other);

				m_xoffset = other.m_xoffset;
				m_yoffset = other.m_yoffset;
			}
			void MouseScrolled::move(MouseScrolled&& other)
			{
				Event::Window::Base::move(std::move(other));

				m_xoffset = other.m_xoffset;
				m_yoffset = other.m_yoffset;
			}
		}
	}
}
