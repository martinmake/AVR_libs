#ifndef _ANNA_LAYER_BASE_H_
#define _ANNA_LAYER_BASE_H_

#include <inttypes.h>
#include <functional>
#include <memory>
#include <assert.h>

#include "anna/shape.h"
#include "anna/hyperparameters.h"

namespace Anna
{
	namespace Layer
	{
		class Base
		{
			private: // MEMBER VARIABLES
				Shape m_shape;
				std::shared_ptr<Hyperparameters> m_hyperparameters;

			public: // STATIC VARIABLES
				static const std::string NAME;
				static const bool        CHANGES_DATA_SHAPE;
				static const bool        IS_OUTPUT;

			public: // CONSTRUCTORS AND DESTRUCTOR
				Base(Shape initial_shape = Shape(0, 0, 0));
				~Base(void);

			public: // MEMBER FUNCTIONS
				void attach_to_neural_network(std::shared_ptr<Hyperparameters> initial_hyperparameters);
				void attach_to_neural_network(const Shape& initial_shape, std::shared_ptr<Hyperparameters> initial_hyperparameters);

			public: // GETTERS
				const Shape& shape(void) const;
			public: // GETTERS FOR STATIC VARIABLES
				virtual const std::string& name              (void) const;
				virtual       bool         changes_data_shape(void) const;
				virtual       bool         is_output         (void) const;
		};

		// GETTERS
		inline const Shape& Base::shape(void) const { return m_shape; }
		// GETTERS FOR STATIC VARIABLES
		inline const std::string& Base::name              (void) const { assert(false && "THIS IS JUST A TEMPLATE"); }
		inline       bool         Base::changes_data_shape(void) const { assert(false && "THIS IS JUST A TEMPLATE"); }
		inline       bool         Base::is_output         (void) const { assert(false && "THIS IS JUST A TEMPLATE"); }
	}
}

#endif
