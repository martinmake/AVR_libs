#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "neural_network.h"
#include "layer.h"

namespace Anna
{
	NeuralNetwork::NeuralNetwork(Cuda::Device& initial_device)
		: m_hyperparameters(std::make_shared<Hyperparameters>()), m_device(initial_device)
	{
	}

	NeuralNetwork::~NeuralNetwork(void)
	{
	}

	void NeuralNetwork::add_layer(const std::string& layer_name, Shape shape)
	{
		if (Layer::is_valid(layer_name))
			add_layer(Layer::construct(layer_name), shape);
		else
		{
			std::cerr << "[NeuralNetwork] add_layer: invalid layer name `" << layer_name << "'" << std::endl;
			exit(1);
		}
	}

	const Tensor& NeuralNetwork::forward(const Tensor& input)
	{
		std::list<std::shared_ptr<Layer::Base>>::iterator previous_layer = m_layers.begin(); previous_layer--;
		std::list<std::shared_ptr<Layer::Base>>::iterator current_layer  = m_layers.begin();

		(*current_layer)->output(input); current_layer++; previous_layer++;
		for (; current_layer != m_layers.end(); current_layer++, previous_layer++)
			(*current_layer)->forward((*previous_layer)->output());

		current_layer--;
		return (*current_layer)->output();
	}
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
