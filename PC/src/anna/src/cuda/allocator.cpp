#include <mutex>

#include "cuda/allocator.h"
#include "cuda/allocator.cuh"

namespace Anna
{
	namespace Cuda
	{
		std::mutex g_mutex;

		void* malloc(uint64_t size)
		{
			void* d_pointer;

			g_mutex.lock();

			d_pointer = cuda_malloc(size);

			g_mutex.unlock();

			return d_pointer;
		}

		void free(void* d_pointer)
		{
			g_mutex.lock();

			cuda_free(d_pointer);

			g_mutex.unlock();
		}

		void memset(void* d_pointer, uint8_t value, uint64_t size)
		{
			cuda_memset(d_pointer, value, size);
		}

		void memcpy(const void* source_pointer, void* destination_pointer, uint64_t size, CopyDirection direction)
		{
			cuda_memcpy(source_pointer, destination_pointer, size, direction);
		}

		uint64_t max_allocation_size(void)
		{
			uint64_t current_max_allocation_size;

			g_mutex.lock();

			current_max_allocation_size = cuda_max_allocation_size();

			g_mutex.unlock();

			return current_max_allocation_size;
		}
	}
}
