#ifndef _ANNA_LAYER_OUTPUT_H_
#define _ANNA_LAYER_OUTPUT_H_

#include <inttypes.h>

#include "anna/layers/base.h"

namespace Anna
{
	namespace Layer
	{
		class Output final : public Base
		{
			public: // STATIC VARIABLES
				static const std::string NAME;

			public: // CONSTRUCTORS AND DESTRUCTOR
				Output(Shape initial_output_shape = Shape(0, 0, 0));
				~Output(void);
		};
	}
}

#endif
