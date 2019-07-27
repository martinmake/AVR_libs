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
			protected: // MEMBER VARIABLES
				Shape m_input_shape,
				      m_output_shape;
				std::shared_ptr<Hyperparameters> m_hyperparameters;

			public: // CONSTRUCTORS AND DESTRUCTOR
				Base(Shape initial_output_shape = Shape::INVALID);
				~Base(void);

			public: // MEMBER FUNCTIONS
				void attach_to_neural_network(const Shape& initial_input_shape, const Shape& initial_output_shape, std::shared_ptr<Hyperparameters> initial_hyperparameters);

			public: // GETTERS
				        const Shape&   input_shape         (void) const;
				        const Shape&   output_shape        (void) const;
				// virtual       uint64_t trainable_parameters(void) const;
			public: // SETTERS
				void input_shape (const Shape& new_input_shape );
				void output_shape(const Shape& new_output_shape);
			public: // GETTERS FOR STATIC VARIABLES
				virtual const std::string& name              (void) const;
				virtual       bool         changes_data_shape(void) const;
				virtual       bool         is_input          (void) const;
				virtual       bool         is_output         (void) const;
		};

		// GETTERS
		inline const Shape& Base::input_shape (void) const { return m_input_shape;  }
		inline const Shape& Base::output_shape(void) const { return m_output_shape; }
		// SETTERS
		inline void Base::input_shape (const Shape& new_input_shape ) { m_input_shape  = new_input_shape;  }
		inline void Base::output_shape(const Shape& new_output_shape) { m_output_shape = new_output_shape; }
		// GETTERS FOR STATIC VARIABLES
		inline const std::string& Base::name              (void) const { assert(false && "THIS IS JUST A TEMPLATE"); }
		inline       bool         Base::changes_data_shape(void) const { assert(false && "THIS IS JUST A TEMPLATE"); }
		inline       bool         Base::is_input          (void) const { assert(false && "THIS IS JUST A TEMPLATE"); }
		inline       bool         Base::is_output         (void) const { assert(false && "THIS IS JUST A TEMPLATE"); }
	}
}

#endif
