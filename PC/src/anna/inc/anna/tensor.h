#ifndef _ANNA_TENSOR_H_
#define _ANNA_TENSOR_H_

#include <inttypes.h>
#include <iostream>
#include "anna/shape.h"

namespace Anna
{
	struct Tensor
	{
		private:
			Shape  m_shape;
			float* m_d_data;

		public:
			Tensor(Shape initial_shape = Shape::INVALID);
			~Tensor(void);

			void copy_from_host(float* h_pointer);
			void copy_to_host  (float* h_pointer) const;

			void  set(Shape location, float value);
			float get(Shape location) const;

			uint64_t shape_to_idx(Shape location) const;

		public: // GETTERS
			const Shape& shape(void) const;
		public: // SETTERS
			void shape(Shape new_shape);

	};

	// GETTERS
	inline const Shape& Tensor::shape(void) const { return m_shape; }
}

#endif
