#include "layers/full_connected.h"

namespace Anna
{
	namespace Layer
	{
		const std::string FullConnected::NAME               = "full_connected";
		const bool        FullConnected::CHANGES_DATA_SHAPE =  true;
		const bool        FullConnected::IS_INPUT           =  false;
		const bool        FullConnected::IS_OUTPUT          =  false;

		FullConnected::FullConnected(Shape initial_shape)
			: Base(initial_shape)
		{
		}

		FullConnected::~FullConnected(void)
		{
		}
	}
}
