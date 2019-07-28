#include "layers/full_connected.h"
#include "cuda/debug.cuh"

namespace Anna
{
	namespace Layer
	{
		static void cuda_forward_kernel(void);
	//	static void cpu_forward_kernel(void); // TODO

		void FullConnected::cuda_forward(const Tensor& input)
		{
			uint64_t input_size  = input.shape().hypervolume();
			uint64_t output_size = m_output.shape().hypervolume();

			m_output = m_biases;

			cudaCall(cudaDeviceSynchronize());
		}
	}
}
