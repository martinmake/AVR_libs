#include "logging.h"

#include "gra/events/window/key.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			Key::Key(int                             initial_key,
			         Input::Window::Keyboard::Action initial_action,
				 Input::Window::Keyboard::Mod    initial_mods)
				: m_key   (initial_key   ),
				  m_action(initial_action),
				  m_mods  (initial_mods  )
			{
			}
			Key::~Key(void)
			{
			}

			void Key::copy(const Key& other)
			{
				Event::Window::Base::copy(other);

				m_key    = other.m_key;
				m_action = other.m_action;
				m_mods   = other.m_mods;
			}
			void Key::move(Key&& other)
			{
				Event::Window::Base::move(std::move(other));

				m_key    = other.m_key;
				m_action = other.m_action;
				m_mods   = other.m_mods;
			}
		}
	}
}
