#ifndef _GPL_EVENT_CATEGORY_BASE_H_
#define _GPL_EVENT_CATEGORY_BASE_H_

#include <string>
#include <functional>

#include <sml/sml.h>
#include <esl/event_categories/base.h>

namespace Gpl
{
	namespace EventCategory
	{
		class Base : public Esl::EventCategory::Base
		{
			protected:
				Base(void);

			DECLARATION_MANDATORY_INTERFACE(Base)
		};

		DEFINITION_MANDATORY(Base, )
	}
}

#endif
