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
			public:
				virtual const std::string& name(void) const = 0;
				virtual operator std::string(void) const { return name(); };
				std::ostream& operator<<(std::ostream& output_stream) const;

			public: // GETTERS
				bool is_handled(void) const;
			public: // SETTERS
				void is_handled(bool new_is_handled);

			protected:
				bool m_is_handled = false;

			DECLARATION_MANDATORY_DERIVE_ONLY(Base)
		};
		inline std::ostream& operator<<(std::ostream& output_stream, const Base& event) { return output_stream << (std::string) event; }

		// GETTERS
		DEFINITION_DEFAULT_GETTER(Base, is_handled, bool)
		// SETTERS
		DEFINITION_DEFAULT_SETTER(Base, is_handled, bool)

		DEFINITION_MANDATORY_DERIVE_ONLY(Base)
	}
}

#endif
