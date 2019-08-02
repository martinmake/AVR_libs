#include <assert.h>
#include <iostream>

#include "layers/base.h"

namespace Anna
{
	namespace Layer
	{
		Base::Base(Shape initial_shape)
			: m_shape(initial_shape)
		{
		}

		Base::~Base(void)
		{
		}

		void Base::attach_to_neural_network(const Shape& initial_input_shape, const Shape& initial_shape, std::shared_ptr<Hyperparameters> initial_hyperparameters)
		{
			assert(initial_shape.is_valid() || m_shape.is_valid());

			if (initial_shape.is_valid()) m_shape           = initial_shape;
			if (true                    ) m_input_shape     = initial_input_shape;
			if (true                    ) m_hyperparameters = initial_hyperparameters;

			init();
		}

		void Base::init(void)
		{
			m_output.shape(m_shape);
			m_error .shape(m_shape);
		}
	}
}
