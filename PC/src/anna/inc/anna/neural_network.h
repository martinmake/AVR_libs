#ifndef _ANNA_NEURAL_NETWORK_H_
#define _ANNA_NEURAL_NETWORK_H_

#include <inttypes.h>
#include <list>
#include <string>

#include "layers/base.h"

#include "anna/cuda/device.ch"
#include "anna/cuda/debug.ch"

namespace Anna
{
	class NeuralNetwork
	{
		private:
			Shape m_input_shape;
			Shape m_output_shape;
		private:
			std::list<Layer::Base> m_layers;
		private:
			Cuda::Device m_device;

		public:
			NeuralNetwork(void);
			~NeuralNetwork(void);

		public:
			void add_layer(const std::string& layer_name);
			void add_layer(const std::string& layer_name, Shape shape);
			void add_layer(Layer::Base& layer);
			NeuralNetwork& operator<<(Layer::Base& layer);

			void forward();
			void train();

		public: // SETTERS
			void input_shape (Shape new_input_shape );
			void output_shape(Shape new_output_shape);

		/*
		public:
			void train(float* dataset);
			float* forward(const float* d_input, float* d_output, const float* d_weights);

		public:
			void set_random_weights(void);
		*/
	};

	inline void           NeuralNetwork::add_layer (Layer::Base& layer) { *this << layer;                          }
	inline NeuralNetwork& NeuralNetwork::operator<<(Layer::Base& layer) { m_layers.push_back(layer); return *this; }

	// SETTERS
	inline void NeuralNetwork::input_shape (Shape new_input_shape ) { m_input_shape  = new_input_shape;  }
	inline void NeuralNetwork::output_shape(Shape new_output_shape) { m_output_shape = new_output_shape; }
}

#endif
