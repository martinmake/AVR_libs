#ifndef _GRA_EVENT_WINDOW_KEY_TYPED_H_
#define _GRA_EVENT_WINDOW_KEY_TYPED_H_

#include <sml/sml.h>

#include "gra/core.h"
#include "gra/events/window/base.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			class KeyTyped : public Event::Window::Base
			{
				public:
					using callback = std::function<void (Event::Window::KeyTyped&)>;

				public:
					KeyTyped(unsigned int initial_keycode);

				public: // GETTERS
					unsigned int keycode(void) const;

				private:
					unsigned int m_keycode;

				DECLARATION_MANDATORY(KeyTyped)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(KeyTyped, keycode, unsigned int)

			DEFINITION_MANDATORY(KeyTyped, other.m_keycode)
		}
	}
}

#endif
