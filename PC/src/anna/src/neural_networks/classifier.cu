#include <float.h>

#include "anna/neural_networks/classifier.h"
#include "anna/cuda/allocator.h"
#include "anna/cuda/debug.cuh"
#include "anna/cuda/std.cuh"

namespace Anna
{
	namespace Cuda
	{
		static Allocator<uint32_t> allocator;
	}
	namespace NeuralNetwork
	{
		__global__ static void cuda_classifie_kernel(
				const float*    output,
				      uint32_t* class_id,
				      uint32_t  count)
		{
			float    max       = output[0];
			uint32_t max_index =        0;
			for (uint32_t i = 1; i < count; i++)
			{
				if (output[i] > max)
				{
					max       = output[i];
					max_index =        i;
				}
			}

			*class_id = max_index;
		}
		uint32_t Classifier::classifie(const Tensor& output) const
		{
			uint32_t count = m_output_shape.hypervolume();
			uint32_t class_id = 0;

			#ifdef USE_CUDA
				uint32_t* cuda_class_id = Cuda::allocator.allocate(1);
				Cuda::allocator.clear(cuda_class_id, 1);

				cuda_classifie_kernel<<<1, 1>>>(
						output.data(),
						cuda_class_id,
						count);
				cudaCall(cudaDeviceSynchronize());
				Cuda::allocator.memcpy(cuda_class_id, &class_id, 1, Cuda::DEVICE_TO_HOST);
			#else
				// TODO: CPU
			#endif

			return class_id;
		}
	}
}
