#include <string>
#include <sstream>
#include <memory>
#include <iomanip>

#include "cuda/allocator.h"
#include "tensor.h"

namespace Anna
{
	namespace Cuda
	{
		Allocator<float> allocator;
	}

	Tensor::Tensor(Shape initial_shape)
		: m_d_data(nullptr)
	{
		shape(initial_shape);
	}
	Tensor::Tensor(const Tensor& other)
		: m_d_data(nullptr)
	{
		*this = other;
	}

	Tensor::~Tensor(void)
	{
		shape(Shape::INVALID);
	}

	void Tensor::copy_from_host(const float* h_pointer)
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

	void Tensor::set_random(float lower_limit, float upper_limit)
	{
		std::vector<float> h_data(m_shape.hypervolume(), 0);

		for (std::vector<float>::iterator it = h_data.begin(); it < h_data.end(); it++)
			*it = (upper_limit - lower_limit) * ((float) std::rand() / RAND_MAX) + lower_limit;

		copy_from_host(h_data);
	}

	void Tensor::clear(void)
	{
		Cuda::allocator.clear(m_d_data, m_shape.hypervolume());
	}

	uint64_t Tensor::shape_to_idx(Shape location) const
	{
		return location.width() +
		       location.height()* m_shape.width() +
		       location.depth() * m_shape.width() * m_shape.height() +
		       location.time()  * m_shape.width() * m_shape.height() * m_shape.depth();
	}

	Tensor::operator std::string() const
	{
		std::stringstream output;
		std::vector<float> h_data(m_shape.hypervolume());

		copy_to_host(h_data);
		std::vector<float>::iterator it = h_data.begin();
		for (uint64_t time = 0; time < m_shape.time(); time++)
		{
			for (uint64_t channel_count = 0; channel_count < m_shape.depth(); channel_count++)
			{
				for (uint64_t height = 0; height < m_shape.height(); height++)
				{
					for (uint64_t width = 0; width < m_shape.width(); width++)
					{
						output << std::fixed << std::setfill(' ') << std::setw(7) << (*it < 0 ? '\0' : '+') << std::setprecision(3) << *it << " ";
						it++;
					}
					if (it != h_data.end()) output << std::endl;
				}
				if (it != h_data.end()) output << std::endl;
			}
			if (it != h_data.end()) output << std::endl;
		}
		output << std::endl;

		return output.str();
	}

	// OPERATORS
	Tensor& Tensor::operator=(const Tensor& other)
	{
		shape(other.shape());
		Cuda::allocator.memcpy(other.d_data(), m_d_data, m_shape.hypervolume(), Cuda::CopyDirection::DEVICE_TO_DEVICE);

		return *this;
	}

	// SETTERS
	void Tensor::shape(Shape new_shape) // TODO: COPY DATA TO THE NEW MEMORY IF ANY
	{
		if (m_shape.hypervolume() == new_shape.hypervolume())
		{
			m_shape = new_shape;
			return;
		}

		m_shape = new_shape;
		if (m_d_data) Cuda::allocator.deallocate(m_d_data);

		if (m_shape.is_valid()) m_d_data = Cuda::allocator.allocate(m_shape.hypervolume());
		else                    m_d_data = nullptr;
	}
}
