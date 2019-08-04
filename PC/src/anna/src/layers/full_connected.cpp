#include <math.h>

#include "layers/full_connected.h"

namespace Anna
{
	namespace Layer
	{
		const std::string FullConnected::NAME                     = "full_connected";
		const bool        FullConnected::CHANGES_DATA_SHAPE       =  true;
		const bool        FullConnected::IS_INPUT                 =  false;
		const bool        FullConnected::IS_OUTPUT                =  false;
		const bool        FullConnected::HAS_TRAINABLE_PARAMETERS =  true;

		FullConnected::FullConnected(Shape initial_output_shape)
			: Base(initial_output_shape)
		{
		}

		FullConnected::~FullConnected(void)
		{
		}

		void FullConnected::init(void)
		{
			Base::init();

			m_weights.shape({ m_input_shape.hypervolume(), 1, m_output.shape().hypervolume() });
			m_weights.set_random(sqrt(2.0 / m_input_shape.hypervolume()));

			m_biases.shape(m_output.shape());
			m_biases.clear();

			m_gradients.shape(m_weights.shape());
		}

		void FullConnected::forward(const std::list<std::shared_ptr<Layer::Base>>::iterator& current_layer)
		{
			m_output = m_biases;

			std::list<std::shared_ptr<Layer::Base>>::iterator previous_layer = current_layer; previous_layer--;

			const Tensor& input = (*previous_layer)->output();
			weigh_input(input);
		}

		void FullConnected::backward(const std::list<std::shared_ptr<Layer::Base>>::reverse_iterator& current_layer, bool update_weights)
		{
			std::list<std::shared_ptr<Layer::Base>>::reverse_iterator next_layer = current_layer; next_layer++;

			if (!(*next_layer)->is_input())
			{
				Tensor& error_back = (*next_layer)->error();
				calculate_error_back(error_back);
			}

			update_biases();

			const Tensor& input = (*next_layer)->output();
			accumulate_gradients(input);
			if (update_weights)
			{
				this->update_weights();
				m_gradients.clear();
			}
		}
	}
}
