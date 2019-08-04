#include "layers/hyperbolic_tangent.h"

namespace Anna
{
	namespace Layer
	{
		const std::string HyperbolicTangent::NAME                     = "hyperbolic_tangent";
		const bool        HyperbolicTangent::CHANGES_DATA_SHAPE       =  false;
		const bool        HyperbolicTangent::IS_INPUT                 =  false;
		const bool        HyperbolicTangent::IS_OUTPUT                =  false;
		const bool        HyperbolicTangent::HAS_TRAINABLE_PARAMETERS =  false;

		HyperbolicTangent::HyperbolicTangent(Shape initial_shape)
			: Base(initial_shape)
		{
		}

		HyperbolicTangent::~HyperbolicTangent(void)
		{
		}

		void HyperbolicTangent::forward(const Tensor& input)
		{
			m_output = input;

			activate();
		}

		void HyperbolicTangent::backward(const Tensor& input, Tensor& error_back, bool update_weights, bool is_next_layer_input)
		{
			(void) input;
			(void) update_weights;
			(void) is_next_layer_input;

			calculate_error_back(error_back);
		}
	}
}
