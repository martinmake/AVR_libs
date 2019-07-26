#ifndef _ANNA_LAYER_OUTPUT_H_
#define _ANNA_LAYER_OUTPUT_H_

#include <inttypes.h>

#include "anna/layers/base.h"

#define ANNA_LAYER_OUTPUT_NAME              "output"
#define ANNA_LAYER_OUTPUT_CHANGES_DATA_SHAPE true

namespace Anna
{
	namespace Layer
	{
		class Output : virtual public Base
		{
			public:
				Output(Shape initial_shape);
				~Output(void);
		};
	}
}

#endif
