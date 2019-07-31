#include <time.h>
#include <iostream>
#include <vector>

#include <anna/neural_network.h>
#include <anna/tensor.h>
#include <anna/cuda/device.cuh>
#include <anna/layers/all.h>

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

	std::vector<Tensor> inputs         (10, nn.input_shape() );
	std::vector<Tensor> desired_outputs(10, nn.output_shape());

	for (Tensor& input : inputs)
		input.set_random(0, 10);
	for (Tensor& desired_output : desired_outputs)
		desired_output.set_random(0, 10);

	nn.hyperparameters().learning_rate(0.05);
	nn.hyperparameters().batch_size(5);

//	nn.dataset(Dataset::Iris("datasets/iris"));
	nn.train(inputs, desired_outputs, 10000);
}
