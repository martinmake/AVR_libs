#include <mutex>

#include "cuda/allocator.h"

namespace Anna
{
	namespace Cuda
	{
		std::mutex mutex;

		void* malloc(uint64_t size)
		{
			void* d_pointer;

			mutex.lock();

			d_pointer = cuda_malloc(size);

			mutex.unlock();

			return d_pointer;
		}

		void free(void* d_pointer)
		{
			mutex.lock();

			cuda_free(d_pointer);

			mutex.unlock();
		}

		void memset(void* d_pointer, uint8_t value, uint64_t size)
		{
			cuda_memset(d_pointer, value, size);
		}

		void memcpy(void* source_pointer, void* destination_pointer, uint64_t size, CopyDirection direction)
		{
			cuda_memcpy(source_pointer, destination_pointer, size, direction);
		}

		uint64_t max_allocation_size(void)
		{
			uint64_t current_max_allocation_size;

			mutex.lock();

			current_max_allocation_size = cuda_max_allocation_size();

			mutex.unlock();

			return current_max_allocation_size;
		}
	}
}
