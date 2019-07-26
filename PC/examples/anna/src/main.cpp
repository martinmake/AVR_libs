#include <time.h>
#include <iostream>

#include <anna/neural_network.h>
#include <anna/layers/all.h>
#include <anna/layer.h> // TMP

int main(void)
{
	using namespace Anna;

	NeuralNetwork nn;

	nn.input_shape ({ 4, 1, 1 });
	nn.output_shape({ 3, 1, 1 });

	nn.add_layer("input"                           );
	nn.add_layer("full_connected",    Shape(7, 1, 1));
	nn.add_layer("hyperbolic_tangent"               );
	nn.add_layer("output"                          );

	// nn.forward();

//	nn.learning_rate(0.05);
//	nn.momentum(0.01);
//	nn.max_epochs(1000);
//	nn.batch_size(10);

//	nn.weight_generation_lower_limit(-0.01);
//	nn.weight_generation_upper_limit(+0.01);
//	nn.weight_generation_seed(time(NULL));

//	nn.dataset(Dataset::Iris("datasets/iris"));
//	nn.train();
}
