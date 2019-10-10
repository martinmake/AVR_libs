#ifndef _GRA_EVENT_WINDOW_RESIZE_H_
#define _GRA_EVENT_WINDOW_RESIZE_H_

#include <sml/sml.h>

#include "gra/core.h"
#include "gra/events/window/base.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			class Resize : public Event::Window::Base
			{
				public:
					using callback = std::function<void (Event::Window::Resize&)>;

				public:
					Resize(int initial_width, int initial_height);

				public: // GETTERS
					int width (void) const;
					int height(void) const;

				private:
					int m_width;
					int m_height;

				DECLARATION_MANDATORY(Resize)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(Resize, width, int)
			DEFINITION_DEFAULT_GETTER(Resize, height, int)

			DEFINITION_MANDATORY(Resize, other.m_width, other.m_height)
		}
	}
}

#endif
