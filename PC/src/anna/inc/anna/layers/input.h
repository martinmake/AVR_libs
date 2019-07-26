#ifndef _ANNA_LAYER_INPUT_H_
#define _ANNA_LAYER_INPUT_H_

#include <inttypes.h>

#include "anna/layers/base.h"

#define ANNA_LAYER_INPUT_NAME              "input"
#define ANNA_LAYER_INPUT_CHANGES_DATA_SHAPE false

namespace Anna
{
	namespace Layer
	{
		class Input : virtual public Base
		{
			public: // STATIC VARIABLES
				static const std::string NAME;
				static const bool        CHANGES_DATA_SHAPE;
				static const bool        IS_OUTPUT;

			public: // CONSTRUCTORS AND DESTRUCTOR
				Input(Shape initial_shape = Shape(0, 0, 0));
				~Input(void);

			public: // GETTERS FOR STATIC VARIABLES
				const std::string& name              (void) const override;
				      bool         changes_data_shape(void) const override;
				      bool         is_output         (void) const override;
		};

		// GETTERS FOR STATIC VARIABLES
		inline const std::string& Input::name              (void) const { return NAME;               }
		inline       bool         Input::changes_data_shape(void) const { return CHANGES_DATA_SHAPE; }
		inline       bool         Input::is_output         (void) const { return IS_OUTPUT;          }
	}
}

#endif
