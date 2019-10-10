#include "logging.h"

#include "gra/events/window/mouse_moved.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			MouseMoved::MouseMoved(float initial_xpos, float initial_ypos)
				: m_xpos(initial_xpos), m_ypos(initial_ypos)
			{
			}
			MouseMoved::~MouseMoved(void)
			{
			}

			void MouseMoved::copy(const MouseMoved& other)
			{
				Event::Window::Base::copy(other);

				m_xpos = other.m_xpos;
				m_ypos = other.m_ypos;
			}
			void MouseMoved::move(MouseMoved&& other)
			{
				Event::Window::Base::move(std::move(other));

				m_xpos = other.m_xpos;
				m_ypos = other.m_ypos;
			}
		}
	}
}
