#ifndef _ANNA_LAYER_OUTPUT_H_
#define _ANNA_LAYER_OUTPUT_H_

#include <inttypes.h>

#include "anna/layers/base.h"

#define ANNA_LAYER_OUTPUT_NAME              "output"
#define ANNA_LAYER_OUTPUT_CHANGES_DATA_SHAPE true

namespace Anna
{
	namespace Layer
	{
		class Output : virtual public Base
		{
			public: // STATIC VARIABLES
				static const std::string NAME;
				static const bool        CHANGES_DATA_SHAPE;
				static const bool        IS_OUTPUT;

			public: // CONSTRUCTORS AND DESTRUCTOR
				Output(Shape initial_shape = Shape(0, 0, 0));
				~Output(void);

			public: // GETTERS FOR STATIC VARIABLES
				const std::string& name              (void) const override;
				      bool         changes_data_shape(void) const override;
				      bool         is_output         (void) const override;
		};

		// GETTERS FOR STATIC VARIABLES
		inline const std::string& Output::name              (void) const { return NAME;               }
		inline       bool         Output::changes_data_shape(void) const { return CHANGES_DATA_SHAPE; }
		inline       bool         Output::is_output         (void) const { return IS_OUTPUT;          }
	}
}

#endif
