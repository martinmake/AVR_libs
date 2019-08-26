#ifndef _GPL_EVENT_CATEGORY_PRIMITIVE_BASE_H_
#define _GPL_EVENT_CATEGORY_PRIMITIVE_BASE_H_

#include "gpl/core.h"
#include "gpl/event_categories/base.h"

namespace Gpl
{
	namespace EventCategory
	{
		namespace Primitive
		{
			class Base : public EventCategory::Base
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
