#ifndef _ANNA_LAYER_FULL_CONNECTED_H_
#define _ANNA_LAYER_FULL_CONNECTED_H_

#include <inttypes.h>

#include "anna/layers/base.h"

namespace Anna
{
	namespace Layer
	{
		class FullConnected : virtual public Base
		{
			public: // STATIC VARIABLES
				static const std::string NAME;
				static const bool        CHANGES_DATA_SHAPE;
				static const bool        IS_INPUT;
				static const bool        IS_OUTPUT;

			public: // CONSTRUCTORS AND DESTRUCTOR
				FullConnected(Shape initial_shape = Shape(0, 0, 0));
				~FullConnected(void);

			public: // GETTERS FOR STATIC VARIABLES
				const std::string& name              (void) const override;
				      bool         changes_data_shape(void) const override;
				      bool         is_input          (void) const override;
				      bool         is_output         (void) const override;
		};

		// GETTERS FOR STATIC VARIABLES
		inline const std::string& FullConnected::name              (void) const { return NAME;               }
		inline       bool         FullConnected::changes_data_shape(void) const { return CHANGES_DATA_SHAPE; }
		inline       bool         FullConnected::is_input          (void) const { return IS_INPUT;           }
		inline       bool         FullConnected::is_output         (void) const { return IS_OUTPUT;          }
	}
}

#endif
