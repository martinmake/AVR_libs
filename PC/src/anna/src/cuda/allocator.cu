#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/allocator.cuh"
#include "cuda/debug.cuh"

namespace Anna
{
	namespace Cuda
	{
		void* cuda_malloc(uint64_t size)
		{
			void* d_pointer;

			cudaCall(cudaMalloc(&d_pointer, size));

			return d_pointer;
		}

		void cuda_free(void* d_pointer)
		{
			cudaCall(cudaFree(d_pointer));
		}

		void cuda_memset(void* d_pointer, uint8_t value, uint64_t size)
		{
			cudaCall(cudaMemset(d_pointer, value, size));
		}

		void cuda_memcpy(const void* source_pointer, void* destination_pointer, uint64_t size, CopyDirection direction)
		{
			switch (direction)
			{
				case HOST_TO_DEVICE:
					cudaCall(cudaMemcpy(destination_pointer, source_pointer, size, cudaMemcpyHostToDevice));
					return;
				case DEVICE_TO_HOST:
					cudaCall(cudaMemcpy(destination_pointer, source_pointer, size, cudaMemcpyDeviceToHost));
					return;
				case DEVICE_TO_DEVICE:
					cudaCall(cudaMemcpy(destination_pointer, source_pointer, size, cudaMemcpyDeviceToDevice));
					return;
			}
		}

		uint64_t cuda_max_allocation_size(void)
		{
			return CU_LIMIT_MALLOC_HEAP_SIZE;
		}
	}
}
