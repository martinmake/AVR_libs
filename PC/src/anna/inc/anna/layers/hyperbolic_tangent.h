#ifndef _ANNA_LAYER_HYPERBOLIC_TANGENT_H_
#define _ANNA_LAYER_HYPERBOLIC_TANGENT_H_

#include <inttypes.h>

#include "anna/layers/base.h"

namespace Anna
{
	namespace Layer
	{
		class HyperbolicTangent final : public Base
		{
			public: // STATIC VARIABLES
				static const std::string NAME;
				static const bool        CHANGES_DATA_SHAPE;
				static const bool        IS_INPUT;
				static const bool        IS_OUTPUT;
				static const bool        HAS_TRAINABLE_PARAMETERS;

			public: // CONSTRUCTORS AND DESTRUCTOR
				HyperbolicTangent(Shape initial_output_shape = Shape::INVALID);
				~HyperbolicTangent(void);

			public:
				void  forward(const Tensor& input) override;
				void backward(const Tensor& input, Tensor& error_back, bool update_weights, bool is_next_layer_input) override;

			private:
				void activate(void);
				void calculate_error_back(Tensor& error_back) const;

			public: // GETTERS FOR STATIC VARIABLES
				const std::string& name                    (void) const override;
				      bool         changes_data_shape      (void) const override;
				      bool         is_input                (void) const override;
				      bool         is_output               (void) const override;
				      bool         has_trainable_parameters(void) const override;
		};

		// GETTERS FOR STATIC VARIABLES
		inline const std::string& HyperbolicTangent::name                    (void) const { return NAME;                     }
		inline       bool         HyperbolicTangent::changes_data_shape      (void) const { return CHANGES_DATA_SHAPE;       }
		inline       bool         HyperbolicTangent::is_input                (void) const { return IS_INPUT;                 }
		inline       bool         HyperbolicTangent::is_output               (void) const { return IS_OUTPUT;                }
		inline       bool         HyperbolicTangent::has_trainable_parameters(void) const { return HAS_TRAINABLE_PARAMETERS; }
	}
}

#endif
