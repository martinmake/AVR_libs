#ifndef _GRA_EVENT_CATEGORY_BASE_H_
#define _GRA_EVENT_CATEGORY_BASE_H_

#include <string>
#include <functional>

#include <sml/sml.h>
#include <esl/event_categories/base.h>

namespace Gra
{
	namespace EventCategory
	{
		class Base : public Esl::EventCategory::Base
		{
			DECLARATION_MANDATORY_INTERFACE(Base)
		};

		DEFINITION_MANDATORY_INTERFACE(Base, )
	}
}

#endif
