#include "anna/kernels/update_biases.cuh"

namespace Anna
{
	namespace Kernel
	{
		__global__ void cuda_update_biases(
					      float* biases,
					const float* error,
					      float  learning_rate,
					      uint64_t neurons_count)
		{
			uint16_t idx = threadIdx.x +
					blockIdx.x * blockDim.x;

			if (idx < neurons_count)
				biases[idx] += error[idx] * learning_rate;
		}
	}
}
