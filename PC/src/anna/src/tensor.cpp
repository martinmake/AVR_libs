#include "cuda/allocator.h"
#include "tensor.h"

namespace Anna
{
	Cuda::Allocator<float> allocator;

	Tensor::Tensor(Shape initial_shape)
	{
		if (initial_shape.is_valid())
			shape(initial_shape);
	}

	Tensor::~Tensor(void)
	{
	}

	void Tensor::copy_from_host(float* h_pointer)
	{
		allocator.memcpy(h_pointer, m_d_data, m_shape.hypervolume(), Cuda::CopyDirection::HOST_TO_DEVICE);
	}

	void Tensor::copy_to_host(float* h_pointer) const
	{
		allocator.memcpy(m_d_data, h_pointer, m_shape.hypervolume(), Cuda::CopyDirection::DEVICE_TO_HOST);
	}

	// SETTERS
	void Tensor::shape(Shape new_shape)
	{
		m_shape = new_shape;

		if (m_d_data) allocator.deallocate(m_d_data);

		m_d_data = allocator.allocate(m_shape.hypervolume());
	}
}
