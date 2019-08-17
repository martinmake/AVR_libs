#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "anna/neural_networks/classifier.h"
#include "anna/layer.h"

namespace Anna
{
	namespace NeuralNetwork
	{
		Classifier::Classifier(void)
		{
		}

		Classifier::~Classifier(void)
		{
		}

		void Classifier::update_accuracy(const Tensor& output, const Tensor& desired_output, float& accuracy)
		{
			uint32_t             classified = classifie(        output);
			uint32_t should_have_classified = classifie(desired_output);

			if (classified == should_have_classified) accuracy++;
		}
	}
}
