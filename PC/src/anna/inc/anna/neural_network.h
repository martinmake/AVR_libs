#ifndef _ANNA_NEURAL_NETWORK_H_
#define _ANNA_NEURAL_NETWORK_H_

#include <inttypes.h>
#include <list>
#include <string>
#include <memory>

#include "layers/all.h"

#include "anna/cuda/device.ch"
#include "anna/cuda/debug.ch"

#include "anna/hyperparameters.h"

namespace Anna
{
	class NeuralNetwork
	{
		private:
			Shape                            m_input_shape;
			Shape                            m_output_shape;
			std::shared_ptr<Hyperparameters> m_hyperparameters;
		private:
			std::list<std::shared_ptr<Layer::Base>> m_layers;
		private:
			Cuda::Device m_device;

		public:
			NeuralNetwork(void);
			NeuralNetwork(const std::string& config_filepath);
			// NeuralNetwork(const Json::Value& config);
			~NeuralNetwork(void);

		public:
			template <typename L>
			void add_layer(        L& layer,      Shape shape = Shape::INVALID);
			template <typename L>
			void add_layer(        L* layer,      Shape shape = Shape::INVALID);
			void add_layer(const std::string& layer_name, Shape shape = Shape::INVALID);

			void forward();
			void train();

			void generate_random_weights(void);

		public: // OPERATORS
			template <typename LayerType>
			NeuralNetwork& operator<<(LayerType* layer);

		public: // GETTERS
			Hyperparameters& hyperparameters(void);
		public: // SETTERS
			void input_shape (Shape new_input_shape );
			void output_shape(Shape new_output_shape);

		/*
			void train(float* dataset);
			float* forward(const float* d_input, float* d_output, const float* d_weights);
		*/
	};

	template <typename L>
	void NeuralNetwork::add_layer(L& layer, Shape shape)
	{
		Shape input_shape  = Shape::INVALID;
		Shape output_shape = Shape::INVALID;

		if (m_layers.size() == 0 && !layer.is_input())
			add_layer(new Layer::Input());

		if (m_layers.size())
			input_shape = (*m_layers.rbegin())->output_shape();
		else
			input_shape = m_input_shape;

		if (!layer.output_shape().is_valid())
		{
			if (shape.is_valid())
				output_shape = shape;
			else
			{
				if (layer.is_input())
				{
					if (m_input_shape.is_valid())
						output_shape = m_input_shape;
					else
					{
						std::cerr << "[NeuralNetwork] add_layer: when using Layer::Input/\"input\" call this->input_shape(Shape shape) before this call"  << std::endl;
						exit(1);
					}
				}
				else if (layer.is_output())
				{
					if (m_output_shape.is_valid())
						output_shape = m_output_shape;
					else
					{
						std::cerr << "[NeuralNetwork] add_layer: when using Layer::Output/\"output\" call this->output_shape(Shape shape) before this call"  << std::endl;
						exit(1);
					}
				}
				else if (!layer.changes_data_shape())
					output_shape = (*m_layers.rbegin())->output_shape();
				else
				{
					std::cerr << "[NeuralNetwork] add_layer: output_shape must be specified for `" << layer.name() << "'"  << std::endl;
					exit(1);
				}
			}
		}

		layer.attach_to_neural_network(input_shape, output_shape, m_hyperparameters);
		m_layers.push_back(std::make_shared<L>(std::move(layer)));
	}

	template <typename L>
	inline void NeuralNetwork::add_layer(L* layer, Shape shape) { add_layer(*layer, shape); }

	// OPERATORS
	template <typename L>
	inline NeuralNetwork& NeuralNetwork::operator<<(L* layer) { add_layer(*layer); return *this; }

	// GETTERS
	inline Hyperparameters& NeuralNetwork::hyperparameters(void) { return *m_hyperparameters; }
	// SETTERS
	inline void NeuralNetwork::input_shape (Shape new_input_shape ) { m_input_shape  = new_input_shape;  }
	inline void NeuralNetwork::output_shape(Shape new_output_shape) { m_output_shape = new_output_shape; }
}

#endif
