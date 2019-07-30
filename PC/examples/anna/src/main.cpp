#include <time.h>
#include <iostream>
#include <vector>

#include <anna/neural_network.h>
#include <anna/tensor.h>
#include <anna/cuda/device.cuh>
#include <anna/layers/all.h>

int main(void)
{
	using namespace Anna;

	Cuda::Device device(0);
	{
		NeuralNetwork nn(device);

		nn.input_shape ({ 4, 1, 1, 1 });
		nn.output_shape({ 3, 1, 1, 1 });

	// 	/* SYNTAX
	// 	nn.add_layer("input"              /*  input   */);
	// 	nn.add_layer("full_connected",    Shape(7, 1, 1));
	// 	nn.add_layer("hyperbolic_tangent" /* constant */);
	// 	nn.add_layer("output"             /*  output  */);
	// 	*/
	//	/* ALTERNATIVE SYNTAX
		nn << new Layer::Input            (          ); // optional
	 	nn << new Layer::FullConnected    (Shape{ 7 });
	 	nn << new Layer::HyperbolicTangent(          );
		nn << new Layer::Output           (          );
	//	*/
		/* ALTERNATIVE SYNTAX
		nn << new Layer::Input            ({ 4, 1, 1 });
		nn << new Layer::FullConnected    ({ 7, 1, 1 });
		nn << new Layer::HyperbolicTangent({ 7, 1, 1 });
		nn << new Layer::FullConnected    ({ 3, 1, 1 });
		*/

		std::vector<Tensor> inputs         (10, nn.input_shape() );
		std::vector<Tensor> desired_outputs(10, nn.output_shape());

		for (Tensor& input : inputs)
			input.set_random(0, 10);
		for (Tensor& desired_output : desired_outputs)
			desired_output.set_random(0, 10);

		nn.hyperparameters().learning_rate(0.05);
		nn.hyperparameters().batch_size(10);

	//	nn.dataset(Dataset::Iris("datasets/iris"));
		nn.train(inputs, desired_outputs);
	}
}
