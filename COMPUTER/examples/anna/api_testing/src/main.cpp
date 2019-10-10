#include <time.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include <anna/neural_networks/base.h>
#include <anna/tensor.h>
#include <anna/cuda/device.cuh>
#include <anna/layers/all.h>
#include <anna/datasets/all.h>

#include "colors.h"

static Anna::Cuda::Device g_device(0);

namespace Anna::Model
{
	class SampleModel : public Model::Base
	{
		public:
			NeuralNetwork::XNet       xnet1;
			NeuralNetwork::XNet       xnet2;
			NeuralNetwork::Classifier output;
	};
}

int main(void)
{
	srand(time(NULL));
	NeuralNetwork::XNet xnet;

	xnet.input_shape ({ 28, 28,  1, 1 });
	xnet.output_shape({  1,  1, 10, 1 });


//	/*
	xnet << Layer::Input      (               );
	xnet << Layer::Convolution(Shape(3, 3, 32));
	xnet << Layer::Relu       (               );
	xnet << Layer::Convolution(Shape(3, 3, 32));
	xnet << Layer::Relu       (               );
	xnet << Layer::Convolution(Shape(3, 3, 32));
	xnet << Layer::Relu       (               );
	xnet << Layer::MaxPooling (Shape(2, 2)    );
	xnet << Layer::Output     (               );
//	*/

	Dataset::Mnist dataset(PROJECT_DATASET_DIRECTORY"/mnist");

	xnet.hyperparameters().learning_rate(0.0001);
	xnet.hyperparameters().batch_size(32);
}
