#ifndef _GRA_EVENT_WINDOW_KEY_H_
#define _GRA_EVENT_WINDOW_KEY_H_

#include <sml/sml.h>

#include "gra/core.h"
#include "gra/events/window/base.h"
#include "gra/inputs/window/keyboard.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			class Key : public Event::Window::Base
			{
				public:
					using callback = std::function<void (Event::Window::Key&)>;

				public:
					Key(int                             initial_key,
					    Input::Window::Keyboard::Action initial_action,
					    Input::Window::Keyboard::Mod    initial_mods);

				public: // GETTERS
					int                             key   (void) const;
					Input::Window::Keyboard::Action action(void) const;
					Input::Window::Keyboard::Mod    mods  (void) const;

				private:
					int                             m_key;
					Input::Window::Keyboard::Action m_action;
					Input::Window::Keyboard::Mod    m_mods;

				DECLARATION_MANDATORY(Key)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(Key, key,    int                            )
			DEFINITION_DEFAULT_GETTER(Key, action, Input::Window::Keyboard::Action)
			DEFINITION_DEFAULT_GETTER(Key, mods,   Input::Window::Keyboard::Mod   )

			DEFINITION_MANDATORY(Key, other.m_key,
			                          other.m_action,
						  other.m_mods)
		}
	}
}

#endif
