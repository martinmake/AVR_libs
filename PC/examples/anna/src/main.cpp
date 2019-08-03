#include <time.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include <anna/neural_networks/classifier.h>
#include <anna/tensor.h>
#include <anna/cuda/device.cuh>
#include <anna/layers/all.h>
#include <anna/datasets/all.h>

#include "colors.h"

void ask_to_proceed(void);

static Anna::Cuda::Device g_device(0);

int main(void)
{
	using namespace Anna;

	NeuralNetwork::Classifier classifier;

	classifier.input_shape ({ 1, 1, 1, 4 });
	classifier.output_shape({ 1, 1, 1, 3 });

// 	SYNTAX
// 	classifier.add_layer("input"              /*  input   */);
// 	classifier.add_layer("full_connected",    Shape(7, 1, 1));
// 	classifier.add_layer("hyperbolic_tangent" /* constant */);
// 	classifier.add_layer("output"             /*  output  */);
// 	SYNTAX
//	ALTERNATIVE SYNTAX
	classifier << new Layer::Input            (                ); // optional
	classifier << new Layer::FullConnected    (Shape{ 1, 1, 7 });
	classifier << new Layer::HyperbolicTangent(                );
	classifier << new Layer::Output           (                );
//	ALTERNATIVE SYNTAX
//	ALTERNATIVE SYNTAX
//	classifier << new Layer::Input            ({ 1, 1, 4 });
//	classifier << new Layer::FullConnected    ({ 1, 1, 7 });
//	classifier << new Layer::HyperbolicTangent({ 1, 1, 7 });
//	classifier << new Layer::FullConnected    ({ 1, 1, 3 });
//	ALTERNATIVE SYNTAX

	Dataset::Iris dataset(PROJECT_DATASET_DIRECTORY"/iris");

	classifier.hyperparameters().learning_rate(0.05);
	classifier.hyperparameters().batch_size(5);

	ask_to_proceed();
	classifier.train(dataset, 10);
	std::cout << "[ACCURACY TRAINING] " << classifier.accuracy_training() * 100 << "%" << std::endl;

	ask_to_proceed();
	classifier.test(dataset);
	std::cout << "[ACCURACY TESTING] "  << classifier.accuracy_testing()  * 100 << "%" << std::endl;
}

void ask_to_proceed(void)
{
	std::cout << Color::BOLD_RED << "Proceed? [Y/n] " << Color::RESET;
	fflush(stdin);
	char c = getchar();
	if (c != '\n') while (getchar() != '\n');

	if (c == 'n' || c == 'N')
	{
		std::cout << Color::BOLD_GREEN << "ABORTING" << Color::RESET << std::endl;
		abort();
	}
}
