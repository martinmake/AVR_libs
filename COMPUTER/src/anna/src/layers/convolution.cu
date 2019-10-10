#include "anna/layers/convolution.h"
#include "anna/kernels/update_biases.cuh"
#include "anna/cuda/debug.cuh"
#include "anna/cuda/std.cuh"

namespace Anna
{
	namespace Kernel
	{
		__global__ static void cuda_weigh_input(
					const float* input,
					      float* output,
					      float* weights,
					      uint64_t   input_count_x,
					      uint64_t   input_count_y,
					      uint64_t   input_count_z,
					      uint64_t  kernel_count_x,
					      uint64_t  kernel_count_y,
					      uint64_t neurons_count)
		{
			uint16_t idx_x      = threadIdx.x + 1 +
			                       blockIdx.x * blockDim.x;
			uint16_t idx_y      = threadIdx.y + 1 +
			                       blockIdx.y * blockDim.y;
			uint16_t neuron_idx = threadIdx.z +
			                       blockIdx.z * blockDim.z;

			if (       idx_x <   input_count_x - 1)
			if (       idx_y <   input_count_y - 1)
			if (neuron_idx   < neurons_count)
			{
				float weighted_sum = 0.0;
				for (int8_t y = -1; y <= 1; y++)
				{
					for (int8_t x = -1; x <= 1; x++)
					{
						weighted_sum += input[idx_x + idx_y * input_count_x + x + y * input_count_x] *
						              weights[(1 + x) + (1 + y) * kernel_count_x + neuron_idx * 9];
					}
				}
				output[idx_x + idx_y * input_count_x + neuron_idx * input_count_x * input_count_y] = weighted_sum;
			//	printf("[%u] [%u, %u] = %f\n", neuron_idx, idx_x, idx_y, weighted_sum);
			}
		}

		__global__ static void cuda_accumulate_gradients(
					      float*   gradients,
					const float*   error,
					const float*   input,
					      uint64_t input_count,
					      uint64_t neurons_count)
		{
			uint16_t  input_idx = threadIdx.x +
			                       blockIdx.x * blockDim.x;
			uint16_t neuron_idx = threadIdx.y +
			                       blockIdx.y * blockDim.y;

			if ( input_idx <  input_count)
			if (neuron_idx < neurons_count)
			{
				float* gradient = gradients  +
				                   input_idx +
				                  neuron_idx * input_count;
				*gradient += error[neuron_idx] * input[input_idx];
			}
		}

		__global__ static void cuda_update_weights(
					      float*   weights,
					const float*   gradients,
					      float    learning_rate,
					      uint64_t input_count,
					      uint64_t neurons_count) {
			uint16_t  input_idx = threadIdx.x +
			                       blockIdx.x * blockDim.x;
			uint16_t neuron_idx = threadIdx.y +
			                       blockIdx.y * blockDim.y;

			if ( input_idx <  input_count)
			if (neuron_idx < neurons_count)
			{
				uint64_t idx = input_idx +
				              neuron_idx * input_count;

				weights[idx] += gradients[idx] * learning_rate;
			}
		}

		__global__ static void cuda_calculate_error_back(
						const float* error,
						      float* error_back,
						const float* weights,

						uint64_t input_count,
						uint64_t neurons_count)
		{
			uint16_t idx = threadIdx.x +
			                blockIdx.x * blockDim.x;

			if (idx < input_count)
			{
				const float* p_error  = error;
				const float* p_weight = weights + idx;
				const float* p_error_end = p_error + neurons_count;

				float err = 0.0;
				while (p_error != p_error_end)
				{
					err      += *p_error * *p_weight;
					p_error  += 1;
					p_weight += input_count;
				}

				error_back[idx] = err;
			}
		}
	}

	namespace Layer
	{
		void Convolution::weigh_input(const Tensor& input)
		{
			uint64_t   input_count_x = m_input_shape.width();
			uint64_t   input_count_y = m_input_shape.height();
			uint64_t   input_count_z = m_input_shape.depth();
			uint64_t  kernel_count_y = m_shape.width();
			uint64_t  kernel_count_z = m_shape.height();
			uint64_t neurons_count   = m_shape.depth();

			#ifdef USE_CUDA
				dim3 block(input_count_x < 32 ?   input_count_x : 32,
				           input_count_y < 32 ?   input_count_y : 32,
				           1);
				dim3 grid((  input_count_x + block.x - 1) / block.x,
				          (  input_count_y + block.y - 1) / block.y,
				          (neurons_count   + block.z - 1) / block.z);

				Kernel::cuda_weigh_input<<<grid, block>>>(
						   input .data(),
						m_output .data(),
						m_weights.data(),

						  input_count_x,
						  input_count_y,
						  input_count_z,
						 kernel_count_y,
						 kernel_count_z,
						neurons_count);

				cudaCall(cudaDeviceSynchronize());
			#else
				// TODO: CPU
			#endif
		}

		void Convolution::accumulate_gradients(const Tensor& input)
		{
			uint64_t input_count   = m_input_shape.hypervolume();
			uint64_t neurons_count =       m_shape.hypervolume();

			#ifdef USE_CUDA
				dim3 block(  input_count < 32 ?   input_count : 32,
				           neurons_count < 32 ? neurons_count : 32);
				dim3 grid((  input_count + block.x - 1) / block.x,
				          (neurons_count + block.y - 1) / block.y);

				Kernel::cuda_accumulate_gradients<<<grid, block>>>(
						m_gradients.data(),
						m_error    .data(),
						  input    .data(),

						  input_count,
						neurons_count);

				cudaCall(cudaDeviceSynchronize());
			#else
				// TODO: CPU
			#endif
		}

		void Convolution::update_biases(void)
		{
			uint64_t neurons_count = m_shape.hypervolume();

			#ifdef USE_CUDA
				dim3 block(neurons_count < 1024 ? neurons_count : 1024);
				dim3 grid((neurons_count + block.y - 1) / block.y);

				Kernel::cuda_update_biases<<<grid, block>>>(
						m_biases.data(),
						m_error .data(),

						m_hyperparameters->learning_rate(),

						neurons_count);

				cudaCall(cudaDeviceSynchronize());
			#else
				// TODO: CPU
			#endif
		}

		void Convolution::update_weights(void)
		{
			uint64_t input_count   = m_input_shape.hypervolume();
			uint64_t neurons_count =       m_shape.hypervolume();

			#ifdef USE_CUDA
				dim3 block(  input_count < 32 ?   input_count : 32,
				           neurons_count < 32 ? neurons_count : 32);
				dim3 grid((  input_count + block.x - 1) / block.x,
				          (neurons_count + block.y - 1) / block.y);

				Kernel::cuda_update_weights<<<grid, block>>>(
						m_weights  .data(),
						m_gradients.data(),

						m_hyperparameters->learning_rate(),

						input_count,
						neurons_count);

				cudaCall(cudaDeviceSynchronize());
			#else
				// TODO: CPU
			#endif
		}

		void Convolution::calculate_error_back(Tensor& error_back)
		{
			uint64_t   input_count =  m_input_shape  .hypervolume();
			uint64_t neurons_count = m_output.shape().hypervolume();

			#ifdef USE_CUDA
				dim3 block(input_count < 1024 ? input_count : 1024);
				dim3 grid((input_count + block.x - 1) / block.x);

				Kernel::cuda_calculate_error_back<<<grid, block>>>(
						m_error     .data(),
						  error_back.data(),
						m_weights   .data(),

						input_count,
						neurons_count);

				cudaCall(cudaDeviceSynchronize());
			#else
				// TODO: CPU
			#endif
		}
	}
}
