#ifndef _ANNA_LAYER_OUTPUT_H_
#define _ANNA_LAYER_OUTPUT_H_

#include <inttypes.h>

#include "anna/layers/base.h"

namespace Anna
{
	namespace Layer
	{
		class Output final : public Base
		{
			public: // STATIC VARIABLES
				static const std::string NAME;
				static const bool        IS_OUTPUT;

			public: // CONSTRUCTORS AND DESTRUCTOR
				Output(Shape initial_output_shape = Shape::INVALID);
				~Output(void);

			public: // GETTERS FOR STATIC VARIABLES
				const std::string& name     (void) const override;
				      bool         is_output(void) const override;
		};

		// GETTERS FOR STATIC VARIABLES
		inline const std::string& Output::name     (void) const { return NAME;      }
		inline       bool         Output::is_output(void) const { return IS_OUTPUT; }
	}
}

#endif
