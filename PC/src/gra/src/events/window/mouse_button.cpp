#include "logging.h"

#include "gra/events/window/mouse_button.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			MouseButton::MouseButton(Input::Window::Mouse::Button initial_button,
			                         Input::Window::Mouse::Action initial_action,
						 Input::Window::Mouse::Mod    initial_mods)
				: m_button(initial_button),
				  m_action(initial_action),
				  m_mods  (initial_mods  )
			{
			}
			MouseButton::~MouseButton(void)
			{
			}

			void MouseButton::copy(const MouseButton& other)
			{
				Event::Window::Base::copy(other);

				m_button = other.m_button;
				m_action = other.m_action;
				m_mods   = other.m_mods;
			}
			void MouseButton::move(MouseButton&& other)
			{
				Event::Window::Base::move(std::move(other));

				m_button = other.m_button;
				m_action = other.m_action;
				m_mods   = other.m_mods;
			}
		}
	}
}
