#include "layers/relu.h"

namespace Anna
{
	namespace Layer
	{
		const std::string Relu::NAME                     = "relu";
		const bool        Relu::CHANGES_DATA_SHAPE       =  false;
		const bool        Relu::IS_INPUT                 =  false;
		const bool        Relu::IS_OUTPUT                =  false;
		const bool        Relu::HAS_TRAINABLE_PARAMETERS =  false;

		Relu::Relu(Shape initial_shape)
			: Base(initial_shape)
		{
		}

		Relu::~Relu(void)
		{
		}

		inline void Relu::forward(const std::list<std::shared_ptr<Base>>::iterator& current_layer)
		{
			std::list<std::shared_ptr<Base>>::iterator previous_layer = current_layer; previous_layer--;

			const Tensor& input = (*previous_layer)->output();
			m_output = input;

			activate();
		}

		inline void Relu::backward(const std::list<std::shared_ptr<Base>>::reverse_iterator& current_layer, bool update_weights)
		{
			(void) update_weights;

			std::list<std::shared_ptr<Base>>::reverse_iterator next_layer = current_layer; next_layer++;

			Tensor& error_back = (*next_layer)->error();
			calculate_error_back(error_back);
		}
	}
}
