#ifndef _GRA_EVENT_WINDOW_MOUSE_BUTTON_H_
#define _GRA_EVENT_WINDOW_MOUSE_BUTTON_H_

#include <sml/sml.h>

#include "gra/core.h"
#include "gra/events/window/base.h"
#include "gra/event_categories/window/input.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			class MouseButton : public Event::Window::Base, public EventCategory::Window::Input
			{
				public:
					using callback = std::function<void (Event::Window::MouseButton&)>;
					enum class Button;

				public:
					MouseButton(Button initial_button, Action initial_action, Mod initial_mods);

				public: // GETTERS
					Button button(void) const;
					Action action(void) const;
					Mod    mods  (void) const;

				public:
					enum class Button
					{
						LEFT   = GLFW_MOUSE_BUTTON_LEFT,
						RIGHT  = GLFW_MOUSE_BUTTON_RIGHT,
						MIDDLE = GLFW_MOUSE_BUTTON_MIDDLE,
						FOUR   = GLFW_MOUSE_BUTTON_4,
						FIVE   = GLFW_MOUSE_BUTTON_5,
						SIX    = GLFW_MOUSE_BUTTON_6,
						SEVEN  = GLFW_MOUSE_BUTTON_7,
						EIGHT  = GLFW_MOUSE_BUTTON_8,
					};

				private:
					Button m_button;
					Action m_action;
					Mod    m_mods;

				DECLARATION_MANDATORY(MouseButton)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(MouseButton, button, MouseButton::Button)
			DEFINITION_DEFAULT_GETTER(MouseButton, action, MouseButton::Action)
			DEFINITION_DEFAULT_GETTER(MouseButton, mods,   MouseButton::Mod   )

			DEFINITION_MANDATORY(MouseButton, other.m_button, other.m_action, other.m_mods)
		}
	}
}

#endif
