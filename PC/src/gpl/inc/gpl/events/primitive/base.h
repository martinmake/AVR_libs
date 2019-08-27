#ifndef _GPL_EVENT_PRIMITIVE_BASE_H_
#define _GPL_EVENT_PRIMITIVE_BASE_H_

#include "gpl/core.h"
#include "gpl/events/base.h"

namespace Gpl
{
	namespace Event
	{
		namespace Primitive
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
