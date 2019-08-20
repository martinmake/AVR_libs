#ifndef _ESL_EVENT_BASE_H_
#define _ESL_EVENT_BASE_H_

#include <string>

#include <sml/sml.h>

namespace Esl
{
	namespace Event
	{
		class Base
		{
			public: // CONSTRUCTORS
				Base(void);

			public: // GETTERS
				bool is_handled(void) const;
			public: // SETTERS
				void is_handled(bool new_is_handled);

			protected:
				bool m_is_handled = false;

			DECLARATION_MANDATORY_INTERFACE(Base)
		};

		// GETTERS
		DEFINITION_DEFAULT_GETTER(Base, is_handled, bool)
		// SETTERS
		DEFINITION_DEFAULT_SETTER(Base, is_handled, bool)

		DEFINITION_MANDATORY(Base, )
	}
}

#endif
