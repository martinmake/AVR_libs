#include <time.h>
#include <iostream>

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

		Tensor input(nn.input_shape());
		Tensor output(nn.output_shape());

		input.set_random(0, 10);
		output = nn.forward(input);

		std::cout << "[INPUT] " << (std::string) input;
		std::cout << "[OUTPUT]" << (std::string) output;

	//	nn.learning_rate(0.05);
	//	nn.momentum(0.01);
	//	nn.max_epochs(1000);
	//	nn.batch_size(10);

	//	nn.dataset(Dataset::Iris("datasets/iris"));
	//	nn.train();
	}
}
