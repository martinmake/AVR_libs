#include "logging.h"

#include "gra/event_categories/window/input.h"

namespace Gra
{
	namespace EventCategory
	{
		namespace Window
		{
			Input::Input(void)
			{
			}

			Input::~Input(void)
			{
			}

			void Input::copy(const Input& other)
			{
				EventCategory::Window::Base::copy(other);
			}
			void Input::move(Input&& other)
			{
				EventCategory::Window::Base::move(std::move(other));
			}
		}
	}
}
