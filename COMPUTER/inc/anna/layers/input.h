#ifndef _ANNA_LAYER_INPUT_H_
#define _ANNA_LAYER_INPUT_H_

#include <inttypes.h>

#include "anna/layers/base.h"

namespace Anna
{
	namespace Layer
	{
		class Input final : public Base
		{
			public: // STATIC VARIABLES
				static const std::string NAME;
				static const bool        CHANGES_DATA_SHAPE;
				static const bool        IS_INPUT;
				static const bool        IS_OUTPUT;
				static const bool        HAS_TRAINABLE_PARAMETERS;

			public: // CONSTRUCTORS AND DESTRUCTOR
				Input(Shape initial_output_shape = Shape::INVALID);
				~Input(void);

			public: // GETTERS FOR STATIC VARIABLES
				const std::string& name                    (void) const override;
				      bool         changes_data_shape      (void) const override;
				      bool         is_input                (void) const override;
				      bool         is_output               (void) const override;
				      bool         has_trainable_parameters(void) const override;
		};

		// GETTERS FOR STATIC VARIABLES
		inline const std::string& Input::name                    (void) const { return NAME;                     }
		inline       bool         Input::changes_data_shape      (void) const { return CHANGES_DATA_SHAPE;       }
		inline       bool         Input::is_input                (void) const { return IS_INPUT;                 }
		inline       bool         Input::is_output               (void) const { return IS_OUTPUT;                }
		inline       bool         Input::has_trainable_parameters(void) const { return HAS_TRAINABLE_PARAMETERS; }
	}
}

#endif
