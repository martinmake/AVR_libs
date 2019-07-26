#include <assert.h>

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

		void Base::attach_to_neural_network(std::shared_ptr<Hyperparameters> initial_hyperparameters)
		{
			assert(m_shape.is_valid());
			m_hyperparameters = initial_hyperparameters;
		}
		void Base::attach_to_neural_network(const Shape& initial_shape, std::shared_ptr<Hyperparameters> initial_hyperparameters)
		{
			m_shape = initial_shape;
			attach_to_neural_network(initial_hyperparameters);
		}
	}
}
