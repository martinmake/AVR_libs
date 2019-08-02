#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "neural_network.h"
#include "layer.h"

namespace Anna
{
	NeuralNetwork::NeuralNetwork(void)
		: m_hyperparameters(std::make_shared<Hyperparameters>())
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
	const Tensor& NeuralNetwork::forward(const std::vector<float>& input)
	{
		static Tensor input_tensor;
		if (input_tensor.shape() != m_input_shape) input_tensor.shape(m_input_shape);

		input_tensor.copy_from_host(input);
		return forward(input_tensor);
	}

	const Tensor& NeuralNetwork::backward(const Tensor& error)
	{
		static uint16_t batch_index = 0;
		bool update_trainable_parameters;

		batch_index++;
		if (batch_index >= m_hyperparameters->batch_size())
		{
			batch_index = 0;
			update_trainable_parameters = true;
		}
		else
			update_trainable_parameters = false;

		std::list<std::shared_ptr<Layer::Base>>::reverse_iterator current_layer  = m_layers.rbegin();
		std::list<std::shared_ptr<Layer::Base>>::reverse_iterator next_layer     = m_layers.rbegin(); next_layer++;

		(*current_layer)->error(error);
		for (; next_layer != m_layers.rend(); current_layer++, next_layer++)
			(*current_layer)->backward((*next_layer)->output(), (*next_layer)->error(), update_trainable_parameters);

		return (*current_layer)->error();
	}

	void NeuralNetwork::train(const Tensor& input, const Tensor& desired_output)
	{
		static Tensor error;
		const Tensor& output = forward(input);

		error  = desired_output;
		error -=         output;

		backward(error);
	}
	void NeuralNetwork::train(const std::vector<float>& input, const std::vector<float>& desired_output)
	{
		static Tensor input_tensor;
		static Tensor desired_output_tensor;

		if (         input_tensor.shape() != m_input_shape )          input_tensor.shape(m_input_shape );
		if (desired_output_tensor.shape() != m_output_shape) desired_output_tensor.shape(m_output_shape);

		input_tensor.copy_from_host(input);
		desired_output_tensor.copy_from_host(desired_output);

		train(input_tensor, desired_output_tensor);
	}

	void NeuralNetwork::train(const std::vector<Tensor>& inputs, const std::vector<Tensor>& desired_outputs, uint64_t epochs, bool verbose)
	{
		int epoch_max_digits = std::to_string(epochs).size();
		// int epoch_max_digits = std::log(epochs);

		uint64_t print_every = inputs.size() / 100;
		if (print_every == 0) print_every = 1;

		for (uint64_t epoch = 0; epoch < epochs; epoch++)
		{
			std::vector<uint64_t> shuffle_indexer(inputs.size());
			for (uint64_t i = 0; i < shuffle_indexer.size(); i++) shuffle_indexer[i] = i;
			std::random_shuffle(shuffle_indexer.begin(), shuffle_indexer.end());


			for (uint64_t i = 0; i < inputs.size(); i++)
			{
				train(inputs[shuffle_indexer[i]], desired_outputs[shuffle_indexer[i]]);

				if (verbose)
				if ((i % print_every) == 0)
					printf("[EPOCH:%*lu/%*lu] %3lu%%\n", epoch_max_digits, epoch + 1, epoch_max_digits, epochs, (i + 1) * 100 / desired_outputs.size());
			}
		}
	}
	void NeuralNetwork::train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& desired_outputs, uint64_t epochs, bool verbose)
	{
		std::vector<Tensor> input_tensors         (         inputs.size(),  m_input_shape);
		std::vector<Tensor> desired_output_tensors(desired_outputs.size(), m_output_shape);

		for(uint64_t i = 0; i < inputs.size(); i++)
		{
			         input_tensors[i].copy_from_host(         inputs[i]);
			desired_output_tensors[i].copy_from_host(desired_outputs[i]);
		}

		train(input_tensors, desired_output_tensors, epochs, verbose);
	}
	void NeuralNetwork::train(const Dataset::Base& dataset, uint64_t epochs, bool verbose)
	{
		const std::vector<Dataset::Item>& training_items = dataset.training_items();
		      std::vector<Tensor>          input_tensors(training_items.size(),  m_input_shape);
		      std::vector<Tensor> desired_output_tensors(training_items.size(), m_output_shape);

		for(uint64_t i = 0; i < training_items.size(); i++)
		{
			         input_tensors[i].copy_from_host(training_items[i]         .input);
			desired_output_tensors[i].copy_from_host(training_items[i].desired_output);
		}

		train(input_tensors, desired_output_tensors, epochs, verbose);
	}
}
