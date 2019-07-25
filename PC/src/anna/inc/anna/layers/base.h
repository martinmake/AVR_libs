#ifndef _ANNA_LAYER_BASE_H_
#define _ANNA_LAYER_BASE_H_

#include <inttypes.h>
#include <functional>

#include "anna/shape.h"

namespace Anna
{
	namespace Layer
	{
		class Base
		{
			private:
				Shape m_shape;

			public:
				Base(Shape initial_shape);
				~Base(void);

			public: // STATIC
		};
	}
}

#endif
