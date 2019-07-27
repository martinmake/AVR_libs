#ifndef _ANNA_TENSOR_H_
#define _ANNA_TENSOR_H_

#include <inttypes.h>
#include <iostream>
#include <vector>

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

		public:
			void copy_from_host(float* h_pointer);
			void copy_from_host(const std::vector<float>& h_pointer);

			void copy_to_host(float* h_pointer) const;
			void copy_to_host(std::vector<float>& h_pointer) const;

		public:
			void  set(Shape location, float value);
			float get(Shape location) const;

		public:
			uint64_t shape_to_idx(Shape location) const;

		public: // OPERATORS
			operator std::string() const;

		public: // GETTERS
			const Shape& shape(void) const;
		public: // SETTERS
			void shape(Shape new_shape);

	};

	// GETTERS
	inline const Shape& Tensor::shape(void) const { return m_shape; }
}

#endif
