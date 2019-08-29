#ifndef _GPL_EVENT_PRIMITIVE_MOUSE_OVER_H_
#define _GPL_EVENT_PRIMITIVE_MOUSE_OVER_H_

#include "gpl/core.h"
#include "gpl/events/primitive/base.h"

namespace Gpl
{
	namespace Event
	{
		namespace Primitive
		{
			class MouseOver : public Event::Primitive::Base
			{
				public:
					using callback = std::function<void (Event::Primitive::MouseOver&)>;

				public:
					MouseOver(void* initial_instance);

				private:
					MouseOver(void);

				DECLARATION_MANDATORY(MouseOver)
			};

			DEFINITION_MANDATORY(MouseOver, )
		}
	}
}

#endif
