#ifndef _ESL_EVENT_BASE_H_
#define _ESL_EVENT_BASE_H_

#include <sml/sml.h>

namespace Esl
{
	namespace Event
	{
		class Base
		{
			private:
				bool m_is_handled = false;

			public: // GETTERS
				bool is_handled(void) const;
			public: // SETTERS
				void is_handled(bool new_is_handled);

			DECLARATION_MANDATORY_DERIVE_ONLY(Base)
		};

		// GETTERS
		DEFINITION_DEFAULT_GETTER(Base, is_handled, bool)
		// SETTERS
		DEFINITION_DEFAULT_SETTER(Base, is_handled, bool)

		DEFINITION_MANDATORY_DERIVE_ONLY(Base)
	}
}

#endif
