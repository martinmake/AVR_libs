#ifndef _ANNA_LAYER_FULL_CONNECTED_H_
#define _ANNA_LAYER_FULL_CONNECTED_H_

#include <inttypes.h>

#include "anna/layers/base.h"

#define ANNA_LAYER_FULL_CONNECTED_NAME              "full_connected"
#define ANNA_LAYER_FULL_CONNECTED_CHANGES_DATA_SHAPE true

namespace Anna
{
	namespace Layer
	{
		class FullConnected : virtual public Base
		{
			public:
				FullConnected(Shape initial_shape);
				~FullConnected(void);
		};
	}
}

#endif
