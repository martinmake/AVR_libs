#ifndef _GRA_EVENT_WINDOW_MOUSE_MOVED_H_
#define _GRA_EVENT_WINDOW_MOUSE_MOVED_H_

#include <sml/sml.h>

#include "gra/core.h"
#include "gra/events/window/base.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			class MouseMoved : public Event::Window::Base
			{
				public:
					using callback = std::function<void (Event::Window::MouseMoved&)>;

				public:
					MouseMoved(float initial_xpos, float initial_ypos);

				public: // GETTERS
					float xpos(void) const;
					float ypos(void) const;

				private:
					float m_xpos;
					float m_ypos;

				DECLARATION_MANDATORY(MouseMoved)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(MouseMoved, xpos, float)
			DEFINITION_DEFAULT_GETTER(MouseMoved, ypos, float)

			DEFINITION_MANDATORY(MouseMoved, other.m_xpos, other.m_ypos)
		}
	}
}

#endif
