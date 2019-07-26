#ifndef _ANNA_LAYER_HYPERBOLIC_TANGENT_H_
#define _ANNA_LAYER_HYPERBOLIC_TANGENT_H_

#include <inttypes.h>

#include "anna/layers/base.h"

#define ANNA_LAYER_HYPERBOLIC_TANGENT_NAME              "hyperbolic_tangent"
#define ANNA_LAYER_HYPERBOLIC_TANGENT_CHANGES_DATA_SHAPE false

namespace Anna
{
	namespace Layer
	{
		class HyperbolicTangent : virtual public Base
		{
			public: // STATIC VARIABLES
				static const std::string NAME;
				static const bool        CHANGES_DATA_SHAPE;
				static const bool        IS_OUTPUT;

			public: // CONSTRUCTORS AND DESTRUCTOR
				HyperbolicTangent(Shape initial_shape = Shape(0, 0, 0));
				~HyperbolicTangent(void);

			public: // GETTERS FOR STATIC VARIABLES
				const std::string& name              (void) const override;
				      bool         changes_data_shape(void) const override;
				      bool         is_output         (void) const override;
		};

		// GETTERS FOR STATIC VARIABLES
		inline const std::string& HyperbolicTangent::name              (void) const { return NAME;               }
		inline       bool         HyperbolicTangent::changes_data_shape(void) const { return CHANGES_DATA_SHAPE; }
		inline       bool         HyperbolicTangent::is_output         (void) const { return IS_OUTPUT;          }
	}
}

#endif
