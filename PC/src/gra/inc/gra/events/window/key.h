#ifndef _GRA_EVENT_WINDOW_KEY_H_
#define _GRA_EVENT_WINDOW_KEY_H_

#include <sml/sml.h>

#include "gra/core.h"
#include "gra/events/window/base.h"
#include "gra/event_categories/window/input.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			class Key : public Event::Window::Base, public EventCategory::Window::Input
			{
				public:
					using callback = std::function<void (Event::Window::Key&)>;

				public:
					Key(int initial_key, Action initial_action, Mod initial_mods);

				public: // GETTERS
					int     key  (void) const;
					Action action(void) const;
					Mod    mods  (void) const;

				private:
					int    m_key;
					Action m_action;
					Mod    m_mods;

				DECLARATION_MANDATORY(Key)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(Key, key,    int        )
			DEFINITION_DEFAULT_GETTER(Key, action, Key::Action)
			DEFINITION_DEFAULT_GETTER(Key, mods,   Key::Mod   )

			DEFINITION_MANDATORY(Key, other.m_key, other.m_action, other.m_mods)
		}
	}
}

#endif
