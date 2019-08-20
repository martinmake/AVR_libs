#ifndef _ESL_EVENT_CATEGORY_BASE_H_
#define _ESL_EVENT_CATEGORY_BASE_H_

#include <sml/sml.h>

namespace Esl
{
	namespace EventCategory
	{
		class Base
		{
			public: // CONSTRUCTORS
				Base(void);

			DECLARATION_MANDATORY_INTERFACE(Base)
		};

		DEFINITION_MANDATORY(Base, )
	}
}

#endif
