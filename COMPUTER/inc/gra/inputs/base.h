#ifndef _GRA_INPUT_BASE_H_
#define _GRA_INPUT_BASE_H_

#include <string>
#include <functional>

#include <sml/sml.h>

namespace Gra
{
	namespace Input
	{
		class Base
		{
			protected:
				Base(void);

			DECLARATION_MANDATORY_INTERFACE(Base)
		};

		DEFINITION_MANDATORY(Base, )
	}
}

#endif
