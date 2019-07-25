#ifndef _ANNA_NEURAL_NETWORK_H_
#define _ANNA_NEURAL_NETWORK_H_

#include <inttypes.h>
#include <list>

#include "layers/base.h"

#include "cuda/device.ch"
#include "cuda/debug.ch"

namespace Anna
{
	class NeuralNetwork
	{
		private:
			std::list<Layer::Base> m_layers;
			Cuda::Device m_device;

		public:
			NeuralNetwork(std::list<Layer::Base>& initial_layers);
			~NeuralNetwork(void);

		/*
		public:
			void train(float* dataset);
			float* forward(const float* d_input, float* d_output, const float* d_weights);

		public:
			void set_random_weights(void);
		*/
	};
}

#endif
