#ifndef _ANNA_LAYER_INPUT_H_
#define _ANNA_LAYER_INPUT_H_

#include <inttypes.h>

#include "anna/layers/base.h"

#define ANNA_LAYER_INPUT_NAME              "input"
#define ANNA_LAYER_INPUT_CHANGES_DATA_SHAPE false

namespace Anna
{
	namespace Layer
	{
		class Input : virtual public Base
		{
			public:
				Input(Shape initial_shape);
				~Input(void);
		};
	}
}

#endif
