#include <math.h>

#include "anna/layers/convolution.h"

namespace Anna
{
	namespace Layer
	{
		const std::string Convolution::NAME                     = "convolution";
		const bool        Convolution::CHANGES_DATA_SHAPE       =  true;
		const bool        Convolution::IS_INPUT                 =  false;
		const bool        Convolution::IS_OUTPUT                =  false;
		const bool        Convolution::HAS_TRAINABLE_PARAMETERS =  true;

		Convolution::Convolution(Shape initial_shape)
			: Base(initial_shape)
		{
		}

		Convolution::~Convolution(void)
		{
		}

		void Convolution::init(void)
		{
			m_output.shape({ m_input_shape.width(), m_input_shape.height(), m_shape.depth() });
			m_error .shape(m_output.shape());

			m_weights.shape(m_shape);
			m_weights.set_random(sqrt(2.0 / m_input_shape.hypervolume()));

			m_biases.shape({ 1, 1, m_shape.depth() });
			m_biases.clear();

			m_gradients.shape(m_weights.shape());
		}

		void Convolution::forward(const std::list<std::shared_ptr<Layer::Base>>::iterator& current_layer)
		{
			m_output = m_biases;

			std::list<std::shared_ptr<Layer::Base>>::iterator previous_layer = current_layer; previous_layer--;

			const Tensor& input = (*previous_layer)->output();
			weigh_input(input);
		}

		void Convolution::backward(const std::list<std::shared_ptr<Layer::Base>>::reverse_iterator& current_layer, bool update_weights)
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
