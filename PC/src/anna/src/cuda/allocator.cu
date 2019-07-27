#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/allocator.cuh"

namespace Anna
{
	namespace Cuda
	{
		uint64_t max_allocation_size(void)
		{
			return CU_LIMIT_MALLOC_HEAP_SIZE;
		}
	}
}
