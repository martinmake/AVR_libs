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
			float* m_data;

		public:
			Tensor(Shape initial_shape = Shape::INVALID);
			Tensor(const Tensor&  other);
			~Tensor(void);

		public:
			void copy_from_host(const float* h_pointer);
			void copy_from_host(const std::vector<float>& h_vector);

			void copy_to_host(float* h_pointer) const;
			void copy_to_host(std::vector<float>& h_vector) const;

		public:
			void  set(Shape location, float value);
			float get(Shape location) const;

		public:
			void seed(float value);

			void set_random(float range);
			void set_random(float lower_limit, float upper_limit);

			void clear(void);

		public:
			uint64_t shape_to_idx(Shape location) const;

		public: // OPERATORS
			Tensor& operator= (const Tensor& other);
			Tensor& operator-=(const Tensor& other);
			operator std::string() const;

		public: // GETTERS
			const Shape& shape(void) const;
			      float* data (void);
			const float* data (void) const;
		public: // SETTERS
			void shape(Shape new_shape);
	};

	inline void Tensor::copy_from_host(const std::vector<float>& h_vector) { copy_from_host(&h_vector[0]); }
	inline void Tensor::copy_to_host(std::vector<float>& h_vector) const { copy_to_host(&h_vector[0]); }

	inline void Tensor::seed(float value) { srand(value); }

	inline void Tensor::set_random(float range) { set_random(-range, +range); }

	// GETTERS
	inline const Shape& Tensor::shape(void) const { return m_shape; }
	inline       float* Tensor::data (void)       { return m_data;  }
	inline const float* Tensor::data (void) const { return m_data;  }
}

#endif
