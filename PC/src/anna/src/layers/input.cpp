#include "layers/input.h"

namespace Anna
{
	namespace Layer
	{
		const std::string Input::NAME               = "input";
		const bool        Input::CHANGES_DATA_SHAPE =  false;
		const bool        Input::IS_INPUT           =  true;
		const bool        Input::IS_OUTPUT          =  false;

		Input::Input(Shape initial_shape)
			: Base(initial_shape)
		{
		}

		Input::~Input(void)
		{
		}
	}
}
