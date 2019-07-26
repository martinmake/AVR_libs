#include "layers/hyperbolic_tangent.h"

namespace Anna
{
	namespace Layer
	{
		const std::string HyperbolicTangent::NAME               = "hyperbolic_tangent";
		const bool        HyperbolicTangent::CHANGES_DATA_SHAPE =  false;
		const bool        HyperbolicTangent::IS_INPUT           =  false;
		const bool        HyperbolicTangent::IS_OUTPUT          =  false;

		HyperbolicTangent::HyperbolicTangent(Shape initial_shape)
			: Base(initial_shape)
		{
		}

		HyperbolicTangent::~HyperbolicTangent(void)
		{
		}
	}
}
