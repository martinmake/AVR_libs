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
				Tensor                           m_input;
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
				        void      forward(const Tensor& input);
			protected:
				virtual void cuda_forward(void);
				virtual void  cpu_forward(void);

			public:
				        void      backward(Tensor& error_back, bool update_trainable_parameters);
			protected:
				virtual void cuda_backward(Tensor& error_back, bool update_trainable_parameters);
				virtual void  cpu_backward(Tensor& error_back, bool update_trainable_parameters);

			public: // GETTERS
				        const Shape&   shape               (void) const;
				        const Tensor&  input               (void) const;
				        const Tensor&  output              (void) const;
				        const Tensor&  error               (void) const;
				              Tensor&  error               (void);
				virtual       uint64_t trainable_parameters(void) const;
				virtual       uint64_t flops               (void) const;
			public: // SETTERS
				void shape (const Shape&  new_shape );
				void input (const Tensor& new_input );
				void output(const Tensor& new_output);
				void error (const Tensor& new_error );
			public: // GETTERS FOR STATIC VARIABLES
				virtual const std::string& name                    (void) const;
				virtual       bool         changes_data_shape      (void) const;
				virtual       bool         is_input                (void) const;
				virtual       bool         is_output               (void) const;
				virtual       bool         has_trainable_parameters(void) const;
		};

		inline void Base::cuda_forward(void) { }
		inline void Base::cpu_forward (void) { }

		inline void Base::cuda_backward(Tensor& error_back, bool update_trainable_parameters) { (void) update_trainable_parameters; error_back = m_error; }
		inline void Base:: cpu_backward(Tensor& error_back, bool update_trainable_parameters) { (void) update_trainable_parameters; error_back = m_error; }

		// GETTERS
		inline const Shape&   Base::shape               (void) const { return m_shape;  }
		inline const Tensor&  Base::input               (void) const { return m_input;  }
		inline const Tensor&  Base::output              (void) const { return m_output; }
		inline const Tensor&  Base::error               (void) const { return m_error;  }
		inline       Tensor&  Base::error               (void)       { return m_error;  }
		inline       uint64_t Base::trainable_parameters(void) const { return 0;        }
		inline       uint64_t Base::flops               (void) const { return 0;        }
		// SETTERS
		inline void Base::shape (const Shape&  new_shape ) { m_shape  = new_shape;  }
		inline void Base::input (const Tensor& new_input ) { m_input  = new_input;  }
		inline void Base::output(const Tensor& new_output) { m_output = new_output; }
		inline void Base::error (const Tensor& new_error ) { m_error  = new_error ; }

		// GETTERS FOR STATIC VARIABLES
		inline const std::string& Base::name                    (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return *new std::string(); }
		inline       bool         Base::changes_data_shape      (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return false;              }
		inline       bool         Base::is_input                (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return false;              }
		inline       bool         Base::is_output               (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return false;              }
		inline       bool         Base::has_trainable_parameters(void) const { assert(false && "THIS IS JUST AN INTERFACE"); return false;              }
	}
}

#endif
