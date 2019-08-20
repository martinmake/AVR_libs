#include "logging.h"

#include "gra/events/window/resize.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			Resize::Resize(int initial_width, int initial_height)
				: m_width(initial_width), m_height(initial_height)
			{
			}
			Resize::~Resize(void)
			{
			}

			void Resize::copy(const Resize& other)
			{
				Event::Window::Base::copy(other);
			}
			void Resize::move(Resize&& other)
			{
				Event::Window::Base::move(std::move(other));

				m_width  = other.m_width;
				m_height = other.m_height;
			}
		}
	}
}
