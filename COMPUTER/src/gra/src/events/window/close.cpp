#include "logging.h"

#include "gra/events/window/close.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			Close::Close(void)
			{
			}
			Close::~Close(void)
			{
			}

			void Close::copy(const Close& other)
			{
				Event::Window::Base::copy(other);
			}
			void Close::move(Close&& other)
			{
				Event::Window::Base::move(std::move(other));
			}
		}
	}
}
