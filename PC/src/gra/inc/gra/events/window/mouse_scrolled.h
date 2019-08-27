#ifndef _GRA_EVENT_WINDOW_MOUSE_SCROLLED_H_
#define _GRA_EVENT_WINDOW_MOUSE_SCROLLED_H_

#include <sml/sml.h>

#include "gra/core.h"
#include "gra/events/window/base.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			class MouseScrolled : public Event::Window::Base
			{
				public:
					using callback = std::function<void (Event::Window::MouseScrolled&)>;

				public:
					MouseScrolled(float initial_xoffset, float initial_yoffset);

				public: // GETTERS
					float xoffset(void) const;
					float yoffset(void) const;

				private:
					float m_xoffset;
					float m_yoffset;

				DECLARATION_MANDATORY(MouseScrolled)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(MouseScrolled, xoffset, float)
			DEFINITION_DEFAULT_GETTER(MouseScrolled, yoffset, float)

			DEFINITION_MANDATORY(MouseScrolled, other.m_xoffset, other.m_yoffset)
		}
	}
}

#endif
