#include "logging.h"

#include "gra/events/window/key_typed.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			KeyTyped::KeyTyped(unsigned int initial_keycode)
				: m_keycode(initial_keycode)
			{
			}
			KeyTyped::~KeyTyped(void)
			{
			}

			void KeyTyped::copy(const KeyTyped& other)
			{
				Event::Window::Base::copy(other);

				m_keycode = other.m_keycode;
			}
			void KeyTyped::move(KeyTyped&& other)
			{
				Event::Window::Base::move(std::move(other));

				m_keycode = other.m_keycode;
			}
		}
	}
}
