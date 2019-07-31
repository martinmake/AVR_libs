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
			assert(      initial_shape.is_valid() ||         m_shape.is_valid());
			assert(initial_input_shape.is_valid() || m_input.shape().is_valid());

			if (      initial_shape.is_valid()) m_shape           = initial_shape;
			if (initial_input_shape.is_valid()) m_input.shape(initial_input_shape);
			if (true                          ) m_hyperparameters = initial_hyperparameters;

			init();
		}

		void Base::init(void)
		{
			m_output.shape(m_shape);
			m_error .shape(m_shape);
		}

		void Base::forward(const Tensor& input)
		{
			m_input = input;

			#ifdef USE_CUDA
				cuda_forward();
			#else
				cpu_forward();
			#endif
		}

		void Base::backward(Tensor& error_back, bool update_trainable_parameters)
		{
			#ifdef USE_CUDA
				cuda_backward(error_back, update_trainable_parameters);
			#else
				cpu_backward(error_back, update_trainable_parameters);
			#endif
		}
	}
}
