#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "neural_network.h"
#include "layer.h"

namespace Anna
{
	NeuralNetwork::NeuralNetwork(void)
		: m_device(0)
	{
	}

	NeuralNetwork::~NeuralNetwork(void)
	{
	}

	void NeuralNetwork::add_layer(const std::string& layer)
	{
		// TODO
	}
	void NeuralNetwork::add_layer(const std::string& layer, Shape shape)
	{
		std::unordered_map<std::string, std::function<Layer::Base*(Shape shape)>>::iterator it = Layer::constructors.find(layer);
		assert(it != Layer::constructors.end());

		std::function<Layer::Base*(Shape shape)> layer_constructor = it->second;
		add_layer(*layer_constructor(shape));
	}
		// m_layers.push_back(*layer_constructor(shape));
}

/*
void NeuralNetwork::train(float* h_dataset)
{
	float*    d_dataset                    = (float*)    cuda_malloc(DATASET_SIZE);
	float*    d_outputs                    = (float*)    cuda_malloc(OUTPUTS_SIZE);
	uint64_t* h_dataset_index_lookup_table = (uint64_t*)      malloc(DATASET_TRAINING_ITEMS_COUNT * sizeof(uint64_t));
	float*    d_weighted_gradients         = (float*)    cuda_malloc(WEIGHTS_SIZE);
	uint32_t* d_confusion_matrix           = (uint32_t*) cuda_malloc(CONFUSION_MATRIX_SIZE);

	cuda_copy_host_to_device(h_dataset, d_dataset, DATASET_SIZE);
	fill_dataset_index_lookup_table(h_dataset_index_lookup_table, DATASET_TRAINING_ITEMS_COUNT);

	cuda_train(d_dataset, h_dataset_index_lookup_table, d_outputs, d_weighted_gradients, m_d_weights, d_confusion_matrix);

	cuda_free(d_dataset);
	cuda_free(d_outputs);
	     free(h_dataset_index_lookup_table);
	cuda_free(d_weighted_gradients);
	cuda_free(d_confusion_matrix);
}

float* NeuralNetwork::forward(const float* d_input)
{
	cuda_forward(d_input, m_d_outputs, m_d_weights);

	return m_d_outputs;
}

void NeuralNetwork::set_random_weights(void)
{
	float* h_weights = (float*) malloc(WEIGHTS_SIZE);
	for (uint64_t i = 0; i < WEIGHTS_COUNT; i++)
		h_weights[i] = (WEIGHTS_INTERVAL_HIGHER - WEIGHTS_INTERVAL_LOWER) * ((float) std::rand() / RAND_MAX) + WEIGHTS_INTERVAL_LOWER;

	cuda_copy_host_to_device(h_weights, m_d_weights, WEIGHTS_SIZE);

	free(h_weights);
}
*/
