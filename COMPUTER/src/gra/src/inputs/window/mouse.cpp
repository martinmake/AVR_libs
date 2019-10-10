#include "logging.h"

#include "gra/inputs/window/mouse.h"

namespace Gra
{
	namespace Input
	{
		namespace Window
		{
			Mouse::Mouse(void)
			{
			}
			Mouse::~Mouse(void)
			{
			}

			void Mouse::copy(const Mouse& other)
			{
				Input::Window::Base::copy(other);
			}
			void Mouse::move(Mouse&& other)
			{
				Input::Window::Base::move(std::move(other));
			}
		}
	}
}
