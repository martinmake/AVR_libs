#ifndef _ANNA_CUDA_ALLOCATOR_CUH_
#define _ANNA_CUDA_ALLOCATOR_CUH_

#include <inttypes.h>

namespace Anna
{
	namespace Cuda
	{
		enum CopyDirection
		{
			HOST_TO_DEVICE, DEVICE_TO_HOST, DEVICE_TO_DEVICE
		};

		extern void* cuda_malloc(uint64_t size);
		extern void cuda_free(void* d_pointer);

		extern void cuda_memset(void* d_pointer, uint8_t value, uint64_t size);

		extern void cuda_memcpy(const void* source_pointer, void* destination_pointer, uint64_t size, CopyDirection direction);

		extern uint64_t cuda_max_allocation_size(void);
	}
}

#endif
