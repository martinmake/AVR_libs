#include "cuda/allocator.h"
#include "tensor.h"

namespace Anna
{
	namespace Cuda
	{
		Allocator<float> allocator;
	}

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
		Cuda::allocator.memcpy(h_pointer, m_d_data, m_shape.hypervolume(), Cuda::CopyDirection::HOST_TO_DEVICE);
	}

	void Tensor::copy_to_host(float* h_pointer) const
	{
		Cuda::allocator.memcpy(m_d_data, h_pointer, m_shape.hypervolume(), Cuda::CopyDirection::DEVICE_TO_HOST);
	}

	void Tensor::set(Shape location, float value)
	{
		uint64_t idx = shape_to_idx(location);

		Cuda::allocator.memcpy(&value, m_d_data + idx, 1, Cuda::CopyDirection::HOST_TO_DEVICE);
	}

	float Tensor::get(Shape location) const
	{
		float value;
		uint64_t idx = shape_to_idx(location);

		Cuda::allocator.memcpy(m_d_data + idx, &value, 1, Cuda::CopyDirection::DEVICE_TO_HOST);

		return value;
	}

	uint64_t Tensor::shape_to_idx(Shape location) const
	{
		return location.width() +
		       location.height()        * m_shape.width() +
		       location.channel_count() * m_shape.width() * m_shape.height() +
		       location.time()          * m_shape.width() * m_shape.height() * m_shape.channel_count();
	}

	// SETTERS
	void Tensor::shape(Shape new_shape)
	{
		m_shape = new_shape;

		if (m_d_data) Cuda::allocator.deallocate(m_d_data);

		m_d_data = Cuda::allocator.allocate(m_shape.hypervolume());
	}
}
