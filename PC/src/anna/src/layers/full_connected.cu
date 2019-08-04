#include "layers/full_connected.h"
#include "cuda/debug.cuh"
#include "cuda/std.cuh"

namespace Anna
{
	namespace Layer
	{
		__global__ static void cuda_weigh_input_kernel(
					const float* input,
					      float* output,
					      float* weights,
					      uint64_t input_count,
					      uint64_t output_count)
		{
			uint16_t idx = threadIdx.x +
			               blockIdx.x * blockDim.x;

			if (idx < output_count)
			{
				const float* p_input     = input;
				      float* p_weights   = weights + idx * input_count;
				const float* p_input_end = input + input_count;

				float sum = 0.0;
				while (p_input != p_input_end)
				{
					sum += *p_input * *p_weights;
					p_input++;
					p_weights++;
				}
				output[idx] += sum;
			}
		}
		void FullConnected::weigh_input(const Tensor& input)
		{
			uint64_t input_count  =  m_input_shape  .hypervolume();
			uint64_t output_count = m_output.shape().hypervolume();

			#ifdef USE_CUDA
				dim3 block(output_count < 1024 ? output_count : 1024);
				dim3 grid((output_count + block.x - 1) / block.x);

				cuda_weigh_input_kernel<<<grid, block>>>(
						  input  .data(),
						m_output .data(),
						m_weights.data(),

						input_count,
						output_count);

				cudaCall(cudaDeviceSynchronize());
			#else
				// TODO: CPU
			#endif
		}

		__global__ static void cuda_accumulate_gradients_kernel(
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
		void FullConnected::accumulate_gradients(const Tensor& input)
		{
			uint64_t input_count   = m_input_shape.hypervolume();
			uint64_t neurons_count =       m_shape.hypervolume();

			#ifdef USE_CUDA
				dim3 block(  input_count < 32 ?   input_count : 32,
				           neurons_count < 32 ? neurons_count : 32);
				dim3 grid((  input_count + block.x - 1) / block.x,
				          (neurons_count + block.y - 1) / block.y);

				cuda_accumulate_gradients_kernel<<<grid, block>>>(
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

		__global__ static void cuda_update_biases_kernel(
					      float* biases,
					const float* error,
					      float  learning_rate,
					      uint64_t neurons_count)
		{
			uint16_t idx = threadIdx.x +
			                blockIdx.x * blockDim.x;

			if (idx < neurons_count)
				biases[idx] += error[idx] * learning_rate;
		}
		void FullConnected::update_biases(void)
		{
			uint64_t neurons_count = m_shape.hypervolume();

			#ifdef USE_CUDA
				dim3 block(neurons_count < 1024 ? neurons_count : 1024);
				dim3 grid((neurons_count + block.y - 1) / block.y);

				cuda_update_biases_kernel<<<grid, block>>>(
						m_biases.data(),
						m_error .data(),

						m_hyperparameters->learning_rate(),

						neurons_count);

				cudaCall(cudaDeviceSynchronize());
			#else
				// TODO: CPU
			#endif
		}

		__global__ static void cuda_update_weights_kernel(
					      float*   weights,
					const float*   gradients,
					      float    learning_rate,
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
				uint64_t idx = input_idx +
				              neuron_idx * input_count;

				weights[idx] += gradients[idx] * learning_rate;
			}
		}
		void FullConnected::update_weights(void)
		{
			uint64_t input_count   = m_input_shape.hypervolume();
			uint64_t neurons_count =       m_shape.hypervolume();

			#ifdef USE_CUDA
				dim3 block(  input_count < 32 ?   input_count : 32,
				           neurons_count < 32 ? neurons_count : 32);
				dim3 grid((  input_count + block.x - 1) / block.x,
				          (neurons_count + block.y - 1) / block.y);

				cuda_update_weights_kernel<<<grid, block>>>(
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

		__global__ static void cuda_calculate_error_back_kernel(
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
		void FullConnected::calculate_error_back(Tensor& error_back)
		{
			uint64_t   input_count =  m_input_shape  .hypervolume();
			uint64_t neurons_count = m_output.shape().hypervolume();

			dim3 block(input_count < 1024 ? input_count : 1024);
			dim3 grid((input_count + block.x - 1) / block.x);

			#ifdef USE_CUDA
				cuda_calculate_error_back_kernel<<<grid, block>>>(
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
