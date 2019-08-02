#include <time.h>
#include <iostream>
#include <vector>

#include <anna/neural_network.h>
#include <anna/tensor.h>
#include <anna/cuda/device.cuh>
#include <anna/layers/all.h>
#include <anna/datasets/all.h>

static Anna::Cuda::Device g_device(0);

int main(void)
{
	using namespace Anna;

	NeuralNetwork nn;

	nn.input_shape ({ 1, 1, 1, 4 });
	nn.output_shape({ 1, 1, 1, 3 });

// 	SYNTAX
// 	nn.add_layer("input"              /*  input   */);
// 	nn.add_layer("full_connected",    Shape(7, 1, 1));
// 	nn.add_layer("hyperbolic_tangent" /* constant */);
// 	nn.add_layer("output"             /*  output  */);
// 	SYNTAX
//	ALTERNATIVE SYNTAX
	nn << new Layer::Input            (                ); // optional
	nn << new Layer::FullConnected    (Shape{ 1, 1, 7 });
	nn << new Layer::HyperbolicTangent(                );
	nn << new Layer::Output           (                );
//	ALTERNATIVE SYNTAX
//	ALTERNATIVE SYNTAX
//	nn << new Layer::Input            ({ 1, 1, 4 });
//	nn << new Layer::FullConnected    ({ 1, 1, 7 });
//	nn << new Layer::HyperbolicTangent({ 1, 1, 7 });
//	nn << new Layer::FullConnected    ({ 1, 1, 3 });
//	ALTERNATIVE SYNTAX

	Dataset::Iris dataset(PROJECT_DATASET_DIRECTORY"/iris");

	nn.hyperparameters().learning_rate(0.05);
	nn.hyperparameters().batch_size(5);

	nn.train(dataset, 10);
}
