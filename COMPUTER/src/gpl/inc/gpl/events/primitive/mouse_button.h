#ifndef _GPL_EVENT_PRIMITIVE_MOUSE_BUTTON_H_
#define _GPL_EVENT_PRIMITIVE_MOUSE_BUTTON_H_

#include "gpl/core.h"
#include "gpl/events/primitive/base.h"
#include "gra/events/window/mouse_button.h"

namespace Gpl
{
	namespace Event
	{
		namespace Primitive
		{
			class MouseButton : public Event::Primitive::Base, public Gra::Event::Window::MouseButton
			{
				public:
					using callback = std::function<void (Event::Primitive::MouseButton&)>;

				public:
					MouseButton(Gra::Input::Window::Mouse::Button initial_button,
					            Gra::Input::Window::Mouse::Action initial_action,
						    Gra::Input::Window::Mouse::Mod    initial_mods,
						    void* initial_instance);

				private:
					MouseButton(void);

				DECLARATION_MANDATORY(MouseButton)
			};

			DEFINITION_MANDATORY(MouseButton, )
		}
	}
}

#endif
