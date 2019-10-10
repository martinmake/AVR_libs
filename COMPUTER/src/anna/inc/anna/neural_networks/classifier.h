#ifndef _ANNA_NEURAL_NETWORK_CLASSIFIER_H_
#define _ANNA_NEURAL_NETWORK_CLASSIFIER_H_

#include "anna/neural_networks/base.h"

namespace Anna
{
	namespace NeuralNetwork
	{
		class Classifier final : public Base
		{
			public:
				Classifier(void);
				// Classifier(const std::string& config_filepath);
				// Classifier(const Json::Value& config);
				~Classifier(void);

			private:
				uint32_t classifie(const Tensor& output) const;

			private:
				void add_output_layer(void) override;
				void update_accuracy(const Tensor& output, const Tensor& desired_output, float& accuracy) override;
		};

		inline void Classifier::add_output_layer(void) { add_layer(new Layer::FullConnected(m_output_shape)); }
	}
}

#endif
