#include "layers/hyperbolic_tangent.h"
#include "cuda/debug.cuh"
#include "cuda/std.cuh"

namespace Anna
{
	namespace Layer
	{
		__global__ static void cuda_activate_kernel(float* data, uint64_t count)
		{
			uint16_t idx = threadIdx.x +
			               blockIdx.x * blockDim.x;

			if (idx < count)
			{
				if      (data[idx] < -20.0) data[idx] = -1;
				else if (data[idx] > +20.0) data[idx] = +1;
				else     data[idx] = tanh(data[idx]);
			}
		}
		void HyperbolicTangent::activate(void)
		{
			uint64_t count = m_shape.hypervolume();

			#ifdef USE_CUDA
				dim3 block(count < 1024 ? count : 1024);
				dim3 grid((count + block.x - 1) / block.x);

				cuda_activate_kernel<<<grid, block>>>(m_output.data(), count);
				cudaCall(cudaDeviceSynchronize());
			#else
				// TODO: CPU
			#endif
		}

		__global__ static void cuda_calculate_error_back_kernel(
				      float* error_back,
				const float* output,
				uint64_t count)
		{
			uint16_t idx = threadIdx.x +
			               blockIdx.x * blockDim.x;

			if (idx < count)
			{
				if      (output[idx] < -20.0) error_back[idx] = -1;
				else if (output[idx] > +20.0) error_back[idx] = +1;
				else
				{
					float cache;
					if      (output[idx] < -20.0) cache = -1;
					else if (output[idx] > +20.0) cache = +1;
					else                          cache = tanh(output[idx]);
					error_back[idx] = 1 - cache * cache;
				}
			}
		}
		void HyperbolicTangent::calculate_error_back(Tensor& error_back) const
		{
			uint64_t count = m_shape.hypervolume();

			#ifdef USE_CUDA
				dim3 block(count < 1024 ? count : 1024);
				dim3 grid((count + block.x - 1) / block.x);

				cuda_calculate_error_back_kernel<<<grid, block>>>(
						error_back.data(),
						m_output  .data(),

						count);
				cudaCall(cudaDeviceSynchronize());
			#else
				// TODO: CPU
			#endif
		}
	}
}
