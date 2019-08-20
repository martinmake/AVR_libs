#ifndef _GRA_EVENT_WINDOW_CLOSE_H_
#define _GRA_EVENT_WINDOW_CLOSE_H_

#include <sml/sml.h>

#include "gra/core.h"
#include "gra/events/window/base.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			class Close : public Event::Window::Base
			{
				public:
					using callback = std::function<void (Event::Window::Close&)>;

				public:
					Close(void);

				DECLARATION_MANDATORY(Close)
			};

			DEFINITION_MANDATORY(Close, )
		}
	}
}

#endif
