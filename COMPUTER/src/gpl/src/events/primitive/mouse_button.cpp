#include "gpl/events/primitive/mouse_button.h"

namespace Gpl
{
	namespace Event
	{
		namespace Primitive
		{
			MouseButton::MouseButton(Gra::Input::Window::Mouse::Button initial_button,
			                         Gra::Input::Window::Mouse::Action initial_action,
						 Gra::Input::Window::Mouse::Mod    initial_mods,
						 void* initial_instance)
				: Event::Primitive::Base(initial_instance),
				  Gra::Event::Window::MouseButton(initial_button,
				                                  initial_action,
				                                  initial_mods)
			{
			}

			MouseButton::~MouseButton(void)
			{
			}

			void MouseButton::copy(const MouseButton& other)
			{
				Event::Primitive::Base::copy(other);
			}
			void MouseButton::move(MouseButton&& other)
			{
				Event::Primitive::Base::move(std::move(other));
			}
		}
	}
}
