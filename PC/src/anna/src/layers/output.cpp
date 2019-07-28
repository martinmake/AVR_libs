#include "layers/output.h"

namespace Anna
{
	namespace Layer
	{
		const std::string Output::NAME = "output";

		Output::Output(Shape initial_shape)
			: Base(initial_shape)
		{
		}

		Output::~Output(void)
		{
		}
	}
}
