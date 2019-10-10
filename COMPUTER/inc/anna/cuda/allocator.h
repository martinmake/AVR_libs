#ifndef _ANNA_CUDA_ALLOCATOR_H_
#define _ANNA_CUDA_ALLOCATOR_H_

#include <inttypes.h>
#include <iostream>

#include "anna/cuda/allocator.cuh"

namespace Anna
{
	namespace Cuda
	{
		extern void* malloc(uint64_t size);
		extern void free(void* d_pointer);
		extern void memset(void* d_pointer, uint8_t value, uint64_t size);
		extern void memcpy(const void* source_pointer, void* destination_pointer, uint64_t size, CopyDirection direction);
		extern uint64_t max_allocation_size(void);

		template <typename T>
		class Allocator
		{
			public:
				Allocator(void);
				~Allocator(void);

			public:
				T* allocate(uint64_t count) const;
				void deallocate(T* d_pointer) const;

				void memcpy(const T* source_pointer, T* destination_pointer, uint64_t count, CopyDirection direction) const;
				void clear(T* pointer, uint64_t count) const;

			public: // GETTERS
				uint64_t max_count(void) const;

		};

		template <typename T>
		Allocator<T>::Allocator(void)
		{
		}

		template <typename T>
		Allocator<T>::~Allocator(void)
		{
		}

		template <typename T>
		T* Allocator<T>::allocate(uint64_t count) const
		{
			return (T*) Cuda::malloc(count * sizeof(T));
		}

		template <typename T>
		void Allocator<T>::deallocate(T* d_pointer) const
		{
			Cuda::free(d_pointer);
		}

		template <typename T>
		void Allocator<T>::memcpy(const T* source_pointer, T* destination_pointer, uint64_t count, CopyDirection direction) const
		{
			Cuda::memcpy(source_pointer, destination_pointer, count * sizeof(T), direction);
		}

		template <typename T>
		void Allocator<T>::clear(T* d_pointer, uint64_t count) const
		{
			Cuda::memset(d_pointer, 0, count * sizeof(T));
		}

		// GETTERS
		template<typename T>
		uint64_t Allocator<T>::max_count(void) const
		{
			return max_allocation_size() / sizeof(T);
		}
	}
}

#endif
