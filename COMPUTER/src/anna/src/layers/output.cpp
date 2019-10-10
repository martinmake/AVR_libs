#include "anna/layers/output.h"

namespace Anna
{
	namespace Layer
	{
		const std::string Output::NAME      = "output";
		const bool        Output::IS_OUTPUT =  true;

		Output::Output(Shape initial_shape)
			: Base(initial_shape)
		{
		}

		Output::~Output(void)
		{
		}
	}
}
