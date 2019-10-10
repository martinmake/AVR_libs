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

	srand(time(NULL));
	NeuralNetwork::Classifier classifier;

	classifier.input_shape ({ 28, 28,  1, 1 });
	classifier.output_shape({  1,  1, 10, 1 });

	/*
	classifier << new Layer::Input              (          );
	classifier << new Layer::FullConnected(Shape(1, 1, 800));
	classifier << new Layer::Relu               (          );
	classifier << new Layer::FullConnected(Shape(1, 1, 800));
	classifier << new Layer::Relu               (          );
	classifier << new Layer::Output             (          );
	*/

//	/*
	classifier << new Layer::Input      (               );
	classifier << new Layer::Convolution(Shape(3, 3, 32));
	classifier << new Layer::Relu       (               );
//	classifier << new Layer::MaxPooling (Shape(2, 2)    );
	classifier << new Layer::Output     (               );
//	*/

	Dataset::Mnist dataset(PROJECT_DATASET_DIRECTORY"/mnist");

	classifier.hyperparameters().learning_rate(0.0001);
	classifier.hyperparameters().batch_size(32);

	ask_to_proceed();
	classifier.train(dataset, 1);
	std::cout << "[ACCURACY TRAINING] " << classifier.accuracy_training() * 100 << "%" << std::endl;
	return 0;

	ask_to_proceed();
	classifier.test(dataset);
	std::cout << "[ACCURACY TESTING] "  << classifier.accuracy_testing()  * 100 << "%" << std::endl;
}

void ask_to_proceed(void)
{
	std::cout << Color::BOLD_RED << "Proceed? [Y/n] " << Color::RESET;
	char c = getchar();
	if (c != '\n') while (getchar() != '\n');

	if (c == 'n' || c == 'N')
	{
		std::cout << Color::BOLD_GREEN << "ABORTING" << Color::RESET << std::endl;
		abort();
	}
}
