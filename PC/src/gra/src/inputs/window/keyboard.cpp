#include "logging.h"

#include "gra/inputs/window/keyboard.h"

namespace Gra
{
	namespace Input
	{
		namespace Window
		{
			Keyboard::Keyboard(void)
			{
			}
			Keyboard::~Keyboard(void)
			{
			}

			void Keyboard::copy(const Keyboard& other)
			{
				Input::Window::Base::copy(other);
			}
			void Keyboard::move(Keyboard&& other)
			{
				Input::Window::Base::move(std::move(other));
			}
		}
	}
}
