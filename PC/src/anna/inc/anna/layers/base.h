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
				Shape                            m_input_shape;
				Shape                            m_output_shape;
				std::shared_ptr<Hyperparameters> m_hyperparameters;
				Tensor                           m_output;

			public: // CONSTRUCTORS AND DESTRUCTOR
				Base(Shape initial_output_shape = Shape::INVALID);
				~Base(void);

			public: // MEMBER FUNCTIONS
				void attach_to_neural_network(const Shape& initial_input_shape, const Shape& initial_output_shape, std::shared_ptr<Hyperparameters> initial_hyperparameters);
				virtual void set_random_trainable_parameters(void);
				virtual void init(void);

			public:
				        void forward     (const Tensor& input);
				virtual void cuda_forward(const Tensor& input);
				virtual void cpu_forward (const Tensor& input);

			public: // GETTERS
				        const Shape&   input_shape         (void) const;
				        const Shape&   output_shape        (void) const;
				        const Tensor&  output              (void) const;
				virtual       uint64_t trainable_parameters(void) const;
				virtual       uint64_t flops               (void) const;
			public: // SETTERS
				void input_shape (const Shape&  new_input_shape );
				void output_shape(const Shape&  new_output_shape);
				void output      (const Tensor& new_output      );
			public: // GETTERS FOR STATIC VARIABLES
				virtual const std::string& name                    (void) const;
				virtual       bool         changes_data_shape      (void) const;
				virtual       bool         is_input                (void) const;
				virtual       bool         is_output               (void) const;
				virtual       bool         has_trainable_parameters(void) const;
		};

		inline void Base::set_random_trainable_parameters(void) { assert(false && "THIS IS JUST AN INTERFACE"); }

		inline void Base::forward(const Tensor& input)
		{
			#ifdef ANNA_USE_CUDA
				cuda_forward(input);
			#else
				cpu_forward(input);
			#endif
		}

		inline void Base::cuda_forward(const Tensor& input) { (void) input; }
		inline void Base::cpu_forward (const Tensor& input) { (void) input; }

		// GETTERS
		inline const Shape&   Base::input_shape         (void) const { return m_input_shape;  }
		inline const Shape&   Base::output_shape        (void) const { return m_output_shape; }
		inline const Tensor&  Base::output              (void) const { return m_output;       }
		inline       uint64_t Base::trainable_parameters(void) const { assert(false && "THIS IS JUST AN INTERFACE"); return 0; }
		inline       uint64_t Base::flops               (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return 0; }
		// SETTERS
		inline void Base::input_shape (const Shape&  new_input_shape ) { m_input_shape  = new_input_shape;  }
		inline void Base::output_shape(const Shape&  new_output_shape) { m_output_shape = new_output_shape; }
		inline void Base::output      (const Tensor& new_output      ) { m_output       = new_output;       }

		// GETTERS FOR STATIC VARIABLES
		inline const std::string& Base::name                    (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return *new std::string(); }
		inline       bool         Base::changes_data_shape      (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return false;              }
		inline       bool         Base::is_input                (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return false;              }
		inline       bool         Base::is_output               (void) const { assert(false && "THIS IS JUST AN INTERFACE"); return false;              }
		inline       bool         Base::has_trainable_parameters(void) const { assert(false && "THIS IS JUST AN INTERFACE"); return false;              }
	}
}

#endif
