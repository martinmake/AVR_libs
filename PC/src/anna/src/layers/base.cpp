#include <assert.h>
#include <iostream>

#include "layers/base.h"

namespace Anna
{
	namespace Layer
	{
		Base::Base(Shape initial_output_shape)
			: m_output_shape(initial_output_shape)
		{
		}

		Base::~Base(void)
		{
		}

		void Base::attach_to_neural_network(const Shape& initial_input_shape, const Shape& initial_output_shape, std::shared_ptr<Hyperparameters> initial_hyperparameters)
		{
			assert(initial_input_shape.is_valid()  || m_input_shape.is_valid() );
			assert(initial_output_shape.is_valid() || m_output_shape.is_valid());

			if (initial_input_shape.is_valid() ) m_input_shape     = initial_input_shape;
			if (initial_output_shape.is_valid()) m_output_shape    = initial_output_shape;
			if (true                           ) m_hyperparameters = initial_hyperparameters;

			init();
		}

		void Base::init(void)
		{
			m_output.shape(m_output_shape);
		}
	}
}
