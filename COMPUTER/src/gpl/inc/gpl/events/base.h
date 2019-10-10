#ifndef _GPL_EVENT_BASE_H_
#define _GPL_EVENT_BASE_H_

#include <string>
#include <functional>

#include <sml/sml.h>
#include <esl/events/base.h>

namespace Gpl
{
	namespace Event
	{
		class Base : public Esl::Event::Base
		{
			protected:
				Base(void);

			DECLARATION_MANDATORY_INTERFACE(Base)
		};

		DEFINITION_MANDATORY(Base, )
	}
}

#endif
