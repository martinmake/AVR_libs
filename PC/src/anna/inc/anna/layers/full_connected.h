#ifndef _ANNA_LAYER_FULL_CONNECTED_H_
#define _ANNA_LAYER_FULL_CONNECTED_H_

#include <inttypes.h>

#include "anna/layers/base.h"
#include "anna/tensor.h"

namespace Anna
{
	namespace Layer
	{
		class FullConnected final : public Base
		{
			public: // STATIC VARIABLES
				static const std::string NAME;
				static const bool        CHANGES_DATA_SHAPE;
				static const bool        IS_INPUT;
				static const bool        IS_OUTPUT;
				static const bool        HAS_TRAINABLE_PARAMETERS;

			private: // MEMBER VARIABLES
				Tensor m_weights;
				Tensor m_biases;
				Tensor m_weighted_gradients;

			public: // CONSTRUCTORS AND DESTRUCTOR
				FullConnected(Shape initial_output_shape = Shape::INVALID);
				~FullConnected(void);

			private:
				void init(void) override;

			private:
				void cuda_forward(void) override;
				void  cpu_forward(void) override;

			public: // GETTERS
				uint64_t trainable_parameters(void) const override;
				uint64_t flops               (void) const override;

			public: // GETTERS FOR STATIC VARIABLES
				const std::string& name                    (void) const override;
				      bool         changes_data_shape      (void) const override;
				      bool         is_input                (void) const override;
				      bool         is_output               (void) const override;
				      bool         has_trainable_parameters(void) const override;
		};

		// GETTERS
		inline uint64_t FullConnected::trainable_parameters(void) const { return m_weights.shape().hypervolume() * m_biases.shape().hypervolume(); }
		inline uint64_t FullConnected::flops               (void) const { return trainable_parameters() * 2;                                       }

		// GETTERS FOR STATIC VARIABLES
		inline const std::string& FullConnected::name                    (void) const { return NAME;                     }
		inline       bool         FullConnected::changes_data_shape      (void) const { return CHANGES_DATA_SHAPE;       }
		inline       bool         FullConnected::is_input                (void) const { return IS_INPUT;                 }
		inline       bool         FullConnected::is_output               (void) const { return IS_OUTPUT;                }
		inline       bool         FullConnected::has_trainable_parameters(void) const { return HAS_TRAINABLE_PARAMETERS; }
	}
}

#endif
