#ifndef _ANNA_LAYER_HYPERBOLIC_TANGENT_H_
#define _ANNA_LAYER_HYPERBOLIC_TANGENT_H_

#include <inttypes.h>

#include "anna/layers/base.h"

#define ANNA_LAYER_HYPERBOLIC_TANGENT_NAME              "hyperbolic_tangent"
#define ANNA_LAYER_HYPERBOLIC_TANGENT_CHANGES_DATA_SHAPE false

namespace Anna
{
	namespace Layer
	{
		class HyperbolicTangent : virtual public Base
		{
			public:
				HyperbolicTangent(Shape initial_shape);
				~HyperbolicTangent(void);
		};
	}
}

#endif
