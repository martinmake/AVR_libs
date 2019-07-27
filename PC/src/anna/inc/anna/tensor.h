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
			float* m_data;

		public:
			Tensor(Shape initial_shape = Shape::INVALID);
			~Tensor(void);

		public: // GETTERS
			Shape& shape(void);
		public: // SETTERS
			void shape(Shape new_shape);

	};

	// GETTERS
	inline Shape& Tensor::shape(void) { return m_shape; }

	// SETTERS
	inline void Tensor::shape(Shape new_shape) { m_shape = new_shape; }
}

#endif
