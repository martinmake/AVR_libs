#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <iostream>

#include <cuda_runtime.h>

#include "neural_network.h"

/*
__device__ uint16_t get_index_of_max_element(const float* d_array, uint16_t size)
{
	float    max_element          = 0.0;
	uint16_t index_of_max_element = 0;

	for (uint16_t i = 0; i < size; i++)
	{
		if (d_array[i] > max_element)
		{
			max_element = d_array[i];
			index_of_max_element = i;
		}
	}

	return index_of_max_element;
}

__global__ void accumulate_weighted_gradients(const float* d_input, float* d_output, const float* d_desired_output, float* d_weighted_gradients, const float* d_weights, uint32_t* d_confusion_matrix)
{
	uint32_t idx = blockIdx.x * BLOCK_DIMX + threadIdx.x;

#include "nn/forward_backward_ready_layers.ch"

	if (idx == 0)
		d_confusion_matrix[get_index_of_max_element(d_desired_output, NEURAL_NETWORK_OUTPUT_COUNT) + get_index_of_max_element(d_output, NEURAL_NETWORK_OUTPUT_COUNT) * NEURAL_NETWORK_OUTPUT_COUNT]++;

	__shared__ float s_gradients[MAX_NEURON_COUNT_IN_ONE_LAYER];
	float* s_gradient = s_gradients + idx;

#include "nn/backward_layers.ch"
}
inline void cuda_accumulate_weighted_gradients(const float* d_input, float* d_output, const float* d_desired_output, float* d_weighted_gradients, const float* d_weights, uint32_t* d_confusion_matrix)
{
	accumulate_weighted_gradients
		<<<(MAX_NEURON_COUNT_IN_ONE_LAYER + BLOCK_DIMX - 1) / BLOCK_DIMX, BLOCK_DIMX>>>
		(d_input, d_output, d_desired_output, d_weighted_gradients, d_weights, d_confusion_matrix);
}

__global__ void test(const float* d_input, float* d_output, const float* d_desired_output, const float* d_weights, uint32_t* d_confusion_matrix)
{
	uint32_t idx = blockIdx.x * BLOCK_DIMX + threadIdx.x;

#include "nn/forward_layers.ch"

	if (idx == 0)
		d_confusion_matrix[get_index_of_max_element(d_desired_output, NEURAL_NETWORK_OUTPUT_COUNT) + get_index_of_max_element(d_output, NEURAL_NETWORK_OUTPUT_COUNT) * NEURAL_NETWORK_OUTPUT_COUNT]++;
}
inline void cuda_test(const float* d_input, float* d_output, const float* d_desired_output, const float* d_weights, uint32_t* d_confusion_matrix)
{
		test<<<(MAX_NEURON_COUNT_IN_ONE_LAYER + BLOCK_DIMX - 1) / BLOCK_DIMX, BLOCK_DIMX>>>
			(d_input, d_output, d_desired_output, d_weights, d_confusion_matrix);
}

__global__ void forward(const float* d_input, float* d_output, const float* d_weights)
{
	uint32_t idx = blockIdx.x * BLOCK_DIMX + threadIdx.x;

#include "nn/forward_layers.ch"
}
inline void cuda_forward(const float* d_input, float* d_output, const float* d_weights)
{
		forward<<<(MAX_NEURON_COUNT_IN_ONE_LAYER + BLOCK_DIMX - 1) / BLOCK_DIMX, BLOCK_DIMX>>>(d_input, d_output, d_weights);
}

__global__ void update_weights(float* d_weights, float* d_weighted_gradients)
{
	uint32_t idx = blockIdx.x * BLOCK_DIMX + threadIdx.x;

	if (idx < WEIGHTS_COUNT)
	{
		d_weights[idx] += d_weighted_gradients[idx];
		d_weighted_gradients[idx] = 0.0;
	}
}
inline void cuda_update_weights(float* d_weights, float* d_weighted_gradients)
{
	update_weights<<<(WEIGHTS_COUNT + BLOCK_DIMX - 1) / BLOCK_DIMX, BLOCK_DIMX>>>(d_weights, d_weighted_gradients);
}

__global__ void print_confusion_matrix(const uint32_t* d_confusion_matrix, const char* d_label)
{
	printf("[CONFUSION_MATRIX:%s:START]\n", d_label);
	for (uint32_t y = 0; y < NEURAL_NETWORK_OUTPUT_COUNT; y++)
	{
		for (uint32_t x = 0; x < NEURAL_NETWORK_OUTPUT_COUNT; x++)
		{
			printf("% 10u ", d_confusion_matrix[y * NEURAL_NETWORK_OUTPUT_COUNT + x]);
		}
		printf("\n");
	}
	printf("[CONFUSION_MATRIX:%s:END]\n", d_label);
}
inline void cuda_print_confusion_matrix(const uint32_t* d_confusion_matrix, const char* h_label)
{
	uint8_t label_count = strlen(h_label) + 1;
	char* d_label = (char*) cuda_malloc(label_count);

	cuda_memcpy_to_device(h_label, d_label, label_count);

	print_confusion_matrix<<<1, 1>>>(d_confusion_matrix, d_label);

	cuda_free(d_label);
}

__global__ void print_accuracy(const uint32_t* d_confusion_matrix, const char* d_label)
{
	float accuracy = 0.0;

	for (uint32_t x = 0; x < NEURAL_NETWORK_OUTPUT_COUNT; x++)
	{
		uint32_t correct_outputs_count;
		uint32_t all_outputs_count_sum = 0;
		for (uint32_t y = 0; y < NEURAL_NETWORK_OUTPUT_COUNT; y++)
		{
			all_outputs_count_sum += d_confusion_matrix[y * NEURAL_NETWORK_OUTPUT_COUNT + x];
			if (y == x)
				correct_outputs_count = d_confusion_matrix[y * NEURAL_NETWORK_OUTPUT_COUNT + x];
		}
		accuracy += (correct_outputs_count / (float) all_outputs_count_sum);
	}

	accuracy /= NEURAL_NETWORK_OUTPUT_COUNT;

	printf("[ACCURACY:%s] %3f%%\n", d_label, accuracy * 100);
}
inline void cuda_print_accuracy(const uint32_t* d_confusion_matrix, const char* h_label)
{
	uint8_t label_count = strlen(h_label) + 1;
	char* d_label = (char*) cuda_malloc(label_count);

	cuda_memcpy_to_device(h_label, d_label, label_count);

	print_accuracy<<<1, 1>>>(d_confusion_matrix, d_label);

	cuda_free(d_label);
}

inline void cuda_print_progress(uint16_t epoch, uint64_t item_index)
{
	printf("[EPOCH:%u/%d] %3u%% DONE\n", epoch, MAX_EPOCHS, (uint8_t) (item_index / ((float) DATASET_ITEMS_COUNT) * 100));
}

void cuda_train(const float* d_dataset, uint64_t* h_dataset_index_lookup_table, float* d_output, float* d_weighted_gradients, float* d_weights, uint32_t* d_confusion_matrix)
{
	for (uint16_t epoch = 1; epoch <= MAX_EPOCHS; epoch++)
	{
		shuffle(h_dataset_index_lookup_table, DATASET_TRAINING_ITEMS_COUNT);
		for (uint64_t i = 0; i < DATASET_TRAINING_ITEMS_COUNT; i++)
		{
			const float* d_input          = d_dataset + h_dataset_index_lookup_table[i] * DATASET_ITEM_COUNT;
			const float* d_desired_output = d_input + NEURAL_NETWORK_INPUT_COUNT;

			cuda_accumulate_weighted_gradients(d_input, d_output, d_desired_output, d_weighted_gradients, d_weights, d_confusion_matrix);

			if (i % BATCH_SIZE == 0)
				cuda_update_weights(d_weights, d_weighted_gradients);

			if (i % 2500 == 0)
				cuda_print_progress(epoch, i);
		}
	}
	cuda_print_confusion_matrix(d_confusion_matrix, "TRAINING");
	cuda_print_accuracy(d_confusion_matrix, "TRAINING");

	cuda_memset(d_confusion_matrix, 0x00, CONFUSION_MATRIX_SIZE);
	d_dataset += DATASET_TRAINING_ITEMS_COUNT * (NEURAL_NETWORK_INPUT_COUNT + NEURAL_NETWORK_OUTPUT_COUNT);
	for (uint64_t i = 0; i < DATASET_TESTING_ITEMS_COUNT; i++)
	{
		const float* d_input = d_dataset + i * (NEURAL_NETWORK_INPUT_COUNT + NEURAL_NETWORK_OUTPUT_COUNT);
		const float* d_desired_output = d_input + NEURAL_NETWORK_INPUT_COUNT;

		cuda_test(d_input, d_output, d_desired_output, d_weights, d_confusion_matrix);
	}
	cuda_print_confusion_matrix(d_confusion_matrix, "TESTING");
	cuda_print_accuracy(d_confusion_matrix, "TESTING");
}
*/
