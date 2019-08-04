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

		inline void HyperbolicTangent::forward(const std::list<std::shared_ptr<Base>>::iterator& current_layer)
		{
			std::list<std::shared_ptr<Base>>::iterator previous_layer = current_layer; previous_layer--;

			const Tensor& input = (*previous_layer)->output();
			m_output = input;

			activate();
		}

		inline void HyperbolicTangent::backward(const std::list<std::shared_ptr<Base>>::reverse_iterator& current_layer, bool update_weights)
		{
			(void) update_weights;

			std::list<std::shared_ptr<Base>>::reverse_iterator next_layer = current_layer; next_layer++;

			Tensor& error_back = (*next_layer)->error();
			calculate_error_back(error_back);
		}
	}
}
