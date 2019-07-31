#ifndef _ANNA_NEURAL_NETWORK_H_
#define _ANNA_NEURAL_NETWORK_H_

#include <inttypes.h>
#include <list>
#include <string>
#include <memory>

#include "anna/cuda/device.cuh"
#include "anna/hyperparameters.h"
#include "anna/tensor.h"
#include "anna/layers/all.h"

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

		public:
			NeuralNetwork(void);
			// NeuralNetwork(const std::string& config_filepath);
			// NeuralNetwork(const Json::Value& config);
			~NeuralNetwork(void);

		public:
			template <typename L> void add_layer(L& layer, Shape shape = Shape::INVALID);
			template <typename L> void add_layer(L* layer, Shape shape = Shape::INVALID);
			void add_layer(const std::string& layer_name, Shape shape = Shape::INVALID);

			const Tensor& forward(const             Tensor& input);
			const Tensor& forward(const std::vector<float>& input);

			const Tensor& backward(const Tensor& error);

			void train(const             Tensor& input, const             Tensor& desired_output);
			void train(const std::vector<float>& input, const std::vector<float>& desired_output);

			void train(const std::vector<            Tensor>& inputs, const std::vector<            Tensor>& desired_outputs, uint64_t epochs = 1, bool verbose = true);
			void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& desired_outputs, uint64_t epochs = 1, bool verbose = true);

		public: // OPERATORS
			template <typename LayerType>
			NeuralNetwork& operator<<(LayerType* layer);

		public: // GETTERS
			Hyperparameters& hyperparameters(void);
			const Shape& input_shape (void) const;
			const Shape& output_shape(void) const;
		public: // SETTERS
			void input_shape (Shape new_input_shape );
			void output_shape(Shape new_output_shape);
	};

	template <typename L>
	void NeuralNetwork::add_layer(L& layer, Shape specific_shape)
	{
		Shape input_shape = Shape::INVALID;
		Shape       shape = Shape::INVALID;

		if (m_layers.size() == 0 && !layer.is_input())
			add_layer(new Layer::Input());

		if (layer.is_output())
		{
			if (m_output_shape.is_valid())
			{
				add_layer(new Layer::FullConnected(m_output_shape));
				return;
			}
			else
			{
				std::cerr << "[NeuralNetwork] add_layer: when using Layer::Output/\"output\" set shape of NeuralNetwork before this call"  << std::endl;
				exit(1);
			}
		}

		if (m_layers.size())
			input_shape = (*m_layers.rbegin())->output().shape();
		else if (m_input_shape.is_valid())
			input_shape = m_input_shape;
		else
		{
			std::cerr << "[NeuralNetwork] add_layer: when using Layer::Intput/\"input\" set input_shape of NeuralNetwork before this call or construct/pass with shape"  << std::endl;
			exit(1);
		}

		if (!layer.shape().is_valid())
		{
			if (specific_shape.is_valid())
				shape = specific_shape;
			else
			{
				if (layer.is_input())
				{
					if (m_input_shape.is_valid())
						shape = m_input_shape;
					else
					{
						std::cerr << "[NeuralNetwork] add_layer: when using Layer::Input/\"input\" call this->input_shape(Shape shape) before this call"  << std::endl;
						exit(1);
					}
				}
				else if (layer.is_output())
				{
					if (m_output_shape.is_valid())
						shape = m_output_shape;
					else
					{
						std::cerr << "[NeuralNetwork] add_layer: when using Layer::Output/\"output\" call this->shape(Shape shape) before this call"  << std::endl;
						exit(1);
					}
				}
				else if (!layer.changes_data_shape())
					shape = (*m_layers.rbegin())->output().shape();
				else
				{
					std::cerr << "[NeuralNetwork] add_layer: shape must be specified for `" << layer.name() << "'"  << std::endl;
					exit(1);
				}
			}
		}

		layer.attach_to_neural_network(input_shape, shape, m_hyperparameters);
		m_layers.push_back(std::make_shared<L>(std::move(layer)));
		// m_layers.emplace_back(&layer);
	}

	template <typename L>
	inline void NeuralNetwork::add_layer(L* layer, Shape shape) { add_layer(*layer, shape); }

	// OPERATORS
	template <typename L>
	inline NeuralNetwork& NeuralNetwork::operator<<(L* layer) { add_layer(*layer); return *this; }

	// GETTERS
	inline Hyperparameters& NeuralNetwork::hyperparameters(void) { return *m_hyperparameters; }
	inline const Shape& NeuralNetwork::input_shape (void) const { return m_input_shape;  }
	inline const Shape& NeuralNetwork::output_shape(void) const { return m_output_shape; }
	// SETTERS
	inline void NeuralNetwork::input_shape (Shape new_input_shape ) { m_input_shape  = new_input_shape;  }
	inline void NeuralNetwork::output_shape(Shape new_output_shape) { m_output_shape = new_output_shape; }
}

#endif
