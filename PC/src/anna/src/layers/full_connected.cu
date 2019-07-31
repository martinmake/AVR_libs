#include "layers/full_connected.h"
#include "cuda/debug.cuh"
#include "cuda/std.cuh"

namespace Anna
{
	namespace Layer
	{
		__global__
		static void cuda_forward_kernel(
					const float* d_input,
					      float* d_output,
					      float* d_weights,
					      uint64_t input_count,
					      uint64_t output_count);

		void FullConnected::cuda_forward(void)
		{
			uint64_t input_count  = m_input .shape().hypervolume();
			uint64_t output_count = m_output.shape().hypervolume();

			m_output = m_biases;

			dim3 block(output_count < 1024 ? output_count : 1024);
			dim3 grid((output_count + block.x - 1) / block.x);

			cuda_forward_kernel<<<grid, block>>>(
					m_input  .data(),
					m_output .data(),
					m_weights.data(),
					input_count,
					output_count);

			cudaCall(cudaDeviceSynchronize());
		}
		__global__
		static void cuda_forward_kernel(
					const float* d_input,
					      float* d_output,
					      float* d_weights,
					      uint64_t input_count,
					      uint64_t output_count)
		{
			uint16_t idx = threadIdx.x +
			               blockIdx.x * blockDim.x;

			if (idx < output_count)
			{
				const float* d_p_input     = d_input;
				      float* d_p_weights   = d_weights + idx * input_count;
				const float* d_p_input_end = d_input + input_count;

				float sum = 0.0;
				while (d_p_input != d_p_input_end)
				{
					sum += *d_p_input * *d_p_weights;
					d_p_input++;
					d_p_weights++;
				}
				d_output[idx] += sum;
			}
		}
	}
}
