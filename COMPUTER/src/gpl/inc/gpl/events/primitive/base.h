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
					Base(void* initial_instance);
					Base(void);

				public:
					template <typename T> T& instance(void);

				private:
					void* m_instance;

				DECLARATION_MANDATORY_INTERFACE(Base)
			};

			template <typename T> T& Base::instance(void) { return *reinterpret_cast<T*>(m_instance); }

			DEFINITION_MANDATORY(Base, other.m_instance)
		}
	}
}

#endif
