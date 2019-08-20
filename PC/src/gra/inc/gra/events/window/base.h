#ifndef _GRA_EVENT_WINDOW_BASE_H_
#define _GRA_EVENT_WINDOW_BASE_H_

#include <string>

#include <sml/sml.h>

#include "gra/core.h"
#include "gra/events/base.h"

namespace Gra
{
	namespace Event
	{
		namespace Window
		{
			class Base : public Event::Base
			{
				protected:
					Base(void);

				DECLARATION_MANDATORY_INTERFACE(Base)
			};

			DEFINITION_MANDATORY(Base, )
		}
	}
}

#endif
