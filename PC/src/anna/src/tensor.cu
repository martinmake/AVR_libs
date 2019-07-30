#include "tensor.h"
#include "tensor.cuh"
#include "cuda/debug.cuh"
#include "cuda/std.cuh"

namespace Anna
{
	namespace Cuda
	{
		__global__
		static void substract_kernel(float* lhs, const float* rhs, uint64_t count);

		void substract(float* lhs, const float* rhs, uint64_t count)
		{
			dim3 block(count < 1024 ? count : 1024);
			dim3 grid((count + block.x - 1) / block.x);

			substract_kernel<<<grid, block>>>(lhs, rhs, count);
			cudaDeviceSynchronize();
		}
		__global__
		static void substract_kernel(float* lhs, const float* rhs, uint64_t count)
		{
			uint64_t idx = threadIdx.x +
			               blockIdx.x * blockDim.x;

			if (idx < count)
				lhs[idx] -= rhs[idx];
		}
	}
}
