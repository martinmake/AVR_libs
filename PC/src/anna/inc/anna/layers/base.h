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
				virtual void set_random_trainable_parameters(void);
				virtual void init(void);

			public: // GETTERS
				        const Shape&   input_shape         (void) const;
				        const Shape&   output_shape        (void) const;
				virtual       uint64_t trainable_parameters(void) const;
				virtual       uint64_t flops               (void) const;
			public: // SETTERS
				void input_shape (const Shape& new_input_shape );
				void output_shape(const Shape& new_output_shape);
			public: // GETTERS FOR STATIC VARIABLES
				virtual const std::string& name                    (void) const;
				virtual       bool         changes_data_shape      (void) const;
				virtual       bool         is_input                (void) const;
				virtual       bool         is_output               (void) const;
				virtual       bool         has_trainable_parameters(void) const;
		};

		inline void Base::set_random_trainable_parameters(void) { assert(false && "THIS IS JUST AN INTERFACE"); }
		inline void Base::init                           (void) { assert(false && "THIS IS JUST AN INTERFACE"); }

		// GETTERS
		inline const Shape&   Base::input_shape         (void) const { return m_input_shape;  }
		inline const Shape&   Base::output_shape        (void) const { return m_output_shape; }
		inline       uint64_t Base::trainable_parameters(void) const { assert(false && "THIS IS JUST AN INTERFACE"); }
		inline       uint64_t Base::flops               (void) const { assert(false && "THIS IS JUST AN INTERFACE"); }
		// SETTERS
		inline void Base::input_shape (const Shape& new_input_shape ) { m_input_shape  = new_input_shape;  }
		inline void Base::output_shape(const Shape& new_output_shape) { m_output_shape = new_output_shape; }

		// GETTERS FOR STATIC VARIABLES
		inline const std::string& Base::name                    (void) const { assert(false && "THIS IS JUST AN INTERFACE"); }
		inline       bool         Base::changes_data_shape      (void) const { assert(false && "THIS IS JUST AN INTERFACE"); }
		inline       bool         Base::is_input                (void) const { assert(false && "THIS IS JUST AN INTERFACE"); }
		inline       bool         Base::is_output               (void) const { assert(false && "THIS IS JUST AN INTERFACE"); }
		inline       bool         Base::has_trainable_parameters(void) const { assert(false && "THIS IS JUST AN INTERFACE"); }
	}
}

#endif
