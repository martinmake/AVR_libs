#ifndef _ANNA_CUDA_ALLOCATOR_H_
#define _ANNA_CUDA_ALLOCATOR_H_

#include <inttypes.h>
#include <iostream>

#include "anna/cuda/allocator.cuh"

namespace Anna
{
	namespace Cuda
	{
		template <typename T>
		class Allocator
		{
			public:
				Allocator(void);
				~Allocator(void);

			public:
				T* allocate(uint64_t count) const;
				void deallocate(T* data, uint64_t count) const;

			public: // GETTERS
				uint64_t max_size(void) const;

		};

		template <typename T>
		Allocator<T>::Allocator(void)
		{
		}

		template <typename T>
		Allocator<T>::~Allocator(void)
		{
		}

		// GETTERS
		template<typename T>
		uint64_t Allocator<T>::max_size(void) const
		{
			return max_allocation_size() / sizeof(T);
		}
	}
}

#endif
