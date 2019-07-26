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
		m_hyperparameters = std::make_shared<Hyperparameters>();
	}

	NeuralNetwork::~NeuralNetwork(void)
	{
	}

	void NeuralNetwork::add_layer(Layer::Base& layer, Shape shape)
	{
		if (m_layers.size() == 0 && !layer.is_input())
			add_layer(*new Layer::Input());

		if (!layer.shape().is_valid())
		{
			if (shape.is_valid())
				layer.shape(shape);
			else
			{
				if (layer.is_input())
				{
					if (m_input_shape.is_valid())
						layer.shape(m_input_shape);
					else
					{
						std::cerr << "[NeuralNetwork] add_layer: when using Layer::Input/\"input\" call this->input_shape(Shape shape) before this call"  << std::endl;
						exit(1);
					}
				}
				else if (layer.is_output())
				{
					if (m_output_shape.is_valid())
						layer.shape(m_output_shape);
					else
					{
						std::cerr << "[NeuralNetwork] add_layer: when using Layer::Output/\"output\" call this->output_shape(Shape shape) before this call"  << std::endl;
						exit(1);
					}
				}
				else if (!layer.changes_data_shape())
					layer.shape(m_layers.rbegin()->shape());
				else
				{
					std::cerr << "[NeuralNetwork] add_layer: shape must be specified for `" << layer.name() << "'"  << std::endl;
					exit(1);
				}
			}
		}

		layer.attach_to_neural_network(m_hyperparameters);
		m_layers.push_back(layer);
	}
	void NeuralNetwork::add_layer(const std::string& layer_name, Shape shape)
	{
		if (Layer::is_valid(layer_name))
			add_layer(*Layer::data(layer_name).constructor(Shape(0, 0, 0)), shape);
		else
		{
			std::cerr << "[NeuralNetwork] add_layer: invalid layer name `" << layer_name << "'" << std::endl;
			exit(1);
		}
	}

	void NeuralNetwork::generate_random_weights(void)
	{
		// TODO
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
