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

			output_shape({ m_output_shape.hypervolume() });

			m_weights.shape({ m_input_shape.hypervolume(), m_output_shape.hypervolume() });
			m_weights.set_random(sqrt(2.0 / m_input_shape.hypervolume()));

			m_weighted_gradients.shape(m_weights.shape());

			m_biases.shape({ m_output_shape.hypervolume() });
			m_biases.clear();
		}
	}
}
