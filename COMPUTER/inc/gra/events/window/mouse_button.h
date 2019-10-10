#ifndef _GRA_EVENT_WINDOW_MOUSE_BUTTON_H_
#define _GRA_EVENT_WINDOW_MOUSE_BUTTON_H_

#include <sml/sml.h>

#include "gra/core.h"
#include "gra/events/window/base.h"
#include "gra/inputs/window/mouse.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			class MouseButton : public Event::Window::Base
			{
				public:
					using callback = std::function<void (Event::Window::MouseButton&)>;
					enum class Button;

				public:
					MouseButton(Input::Window::Mouse::Button initial_button,
					            Input::Window::Mouse::Action initial_action,
						    Input::Window::Mouse::Mod    initial_mods);

				public: // GETTERS
					Input::Window::Mouse::Button button(void) const;
					Input::Window::Mouse::Action action(void) const;
					Input::Window::Mouse::Mod    mods  (void) const;

				private:
					Input::Window::Mouse::Button m_button;
					Input::Window::Mouse::Action m_action;
					Input::Window::Mouse::Mod    m_mods;

				DECLARATION_MANDATORY(MouseButton)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(MouseButton, button, Input::Window::Mouse::Button)
			DEFINITION_DEFAULT_GETTER(MouseButton, action, Input::Window::Mouse::Action)
			DEFINITION_DEFAULT_GETTER(MouseButton, mods,   Input::Window::Mouse::Mod   )

			DEFINITION_MANDATORY(MouseButton, other.m_button,
			                                  other.m_action,
							  other.m_mods)
		}
	}
}

#endif
