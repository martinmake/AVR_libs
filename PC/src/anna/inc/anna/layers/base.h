#ifndef _ANNA_LAYER_BASE_H_
#define _ANNA_LAYER_BASE_H_

#include <inttypes.h>
#include <functional>
#include <memory>
#include <assert.h>

#include "anna/shape.h"
#include "anna/tensor.h"
#include "anna/hyperparameters.h"

namespace Anna
{
	namespace Layer
	{
		class Base
		{
			protected: // MEMBER VARIABLES
				std::shared_ptr<Hyperparameters> m_hyperparameters;
				Shape                            m_shape;
				Shape                            m_input_shape;
				Tensor                           m_output;
				Tensor                           m_error;

			public: // CONSTRUCTORS AND DESTRUCTOR
				Base(Shape initial_shape = Shape::INVALID);
				~Base(void);

			public: // MEMBER FUNCTIONS
				void attach_to_neural_network(const Shape& initial_input_shape, const Shape& initial_shape, std::shared_ptr<Hyperparameters> initial_hyperparameters);
			protected:
				virtual void init(void);

			public:
				virtual void  forward(const Tensor& input);
				virtual void backward(const Tensor& input, Tensor& error_back, bool update_trainable_parameters);

			public: // GETTERS
				        const Shape&   shape               (void) const;
				        const Shape&   input_shape         (void) const;
				        const Tensor&  output              (void) const;
				        const Tensor&  error               (void) const;
				              Tensor&  error               (void);
				virtual       uint64_t trainable_parameters(void) const;
				virtual       uint64_t flops               (void) const;
			public: // SETTERS
				void shape       (const Shape&  new_shape      );
				void input_shape (const Shape&  new_input_shape);
				void output      (const Tensor& new_output     );
				void error       (const Tensor& new_error      );
			public: // GETTERS FOR STATIC VARIABLES
				virtual const std::string& name                    (void) const;
				virtual       bool         changes_data_shape      (void) const;
				virtual       bool         is_input                (void) const;
				virtual       bool         is_output               (void) const;
				virtual       bool         has_trainable_parameters(void) const;
		};

		inline void Base:: forward(const Tensor& input) { (void) input; }
		inline void Base::backward(const Tensor& input, Tensor& error_back, bool update_trainable_parameters) { (void) input; (void) update_trainable_parameters; error_back = m_error; }

		// GETTERS
		inline const Shape&   Base::shape               (void) const { return m_shape;  }
		inline const Tensor&  Base::output              (void) const { return m_output; }
		inline const Tensor&  Base::error               (void) const { return m_error;  }
		inline       Tensor&  Base::error               (void)       { return m_error;  }
		inline       uint64_t Base::trainable_parameters(void) const { return 0;        }
		inline       uint64_t Base::flops               (void) const { return 0;        }
		// SETTERS
		inline void Base::shape      (const Shape&  new_shape      ) { m_shape       = new_shape;       }
		inline void Base::input_shape(const Shape&  new_input_shape) { m_input_shape = new_input_shape; }
		inline void Base::output     (const Tensor& new_output     ) { m_output      = new_output;      }
		inline void Base::error      (const Tensor& new_error      ) { m_error       = new_error ;      }

		// GETTERS FOR STATIC VARIABLES
		inline const std::string& Base::name                    (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return *new std::string(); }
		inline       bool         Base::changes_data_shape      (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return false;              }
		inline       bool         Base::is_input                (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return false;              }
		inline       bool         Base::is_output               (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return false;              }
		inline       bool         Base::has_trainable_parameters(void) const { assert(false && "THIS IS JUST AN INTERFACE"); return false;              }
	}
}

#endif
