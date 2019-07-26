#ifndef _ANNA_NEURAL_NETWORK_H_
#define _ANNA_NEURAL_NETWORK_H_

#include <inttypes.h>
#include <list>
#include <string>
#include <memory>

#include "layers/base.h"

#include "anna/cuda/device.ch"
#include "anna/cuda/debug.ch"

#include "anna/hyperparameters.h"

namespace Anna
{
	class NeuralNetwork
	{
		private: Shape                            m_input_shape;
			Shape                            m_output_shape;
			std::shared_ptr<Hyperparameters> m_hyperparameters;
		private:
			std::list<Layer::Base> m_layers;
		private:
			Cuda::Device m_device;

		public:
			NeuralNetwork(void);
			NeuralNetwork(const std::string& config_filepath);
			// NeuralNetwork(const Json::Value& config);
			~NeuralNetwork(void);

		public:
			void add_layer(      Layer::Base& layer,      Shape shape = Shape(0, 0, 0));
			void add_layer(const std::string& layer_name, Shape shape = Shape(0, 0, 0));

			void forward();
			void train();

			void generate_random_weights(void);

		public: // OPERATORS
			NeuralNetwork& operator<<(Layer::Base& layer);

		public: // GETTERS
			Hyperparameters& hyperparameters(void);
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

	// OPERATORS
	inline NeuralNetwork& NeuralNetwork::operator<<(Layer::Base& layer) { add_layer(layer); return *this; }

	// GETTERS
	inline Hyperparameters& NeuralNetwork::hyperparameters(void) { return *m_hyperparameters; }
	// SETTERS
	inline void NeuralNetwork::input_shape (Shape new_input_shape ) { m_input_shape  = new_input_shape;  }
	inline void NeuralNetwork::output_shape(Shape new_output_shape) { m_output_shape = new_output_shape; }
}

#endif
