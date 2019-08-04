#ifndef _ANNA_NEURAL_NETWORK_BASE_H_
#define _ANNA_NEURAL_NETWORK_BASE_H_

#include <inttypes.h>
#include <list>
#include <string>
#include <memory>

#include "anna/cuda/device.cuh"
#include "anna/hyperparameters.h"
#include "anna/tensor.h"
#include "anna/layers/all.h"
#include "anna/datasets/all.h"

namespace Anna
{
	namespace NeuralNetwork
	{
		class Base
		{
			protected:
				Shape                            m_input_shape;
				Shape                            m_output_shape;
				std::shared_ptr<Hyperparameters> m_hyperparameters;
			protected:
				float m_accuracy_training;
				float m_accuracy_testing;
			protected:
				std::list<std::shared_ptr<Layer::Base>> m_layers;

			public:
				Base(void);
				// Base(const std::string& config_filepath);
				// Base(const Json::Value& config);
				~Base(void);

			public:
				template <typename L> void add_layer(L& layer, Shape shape = Shape::INVALID);
				template <typename L> void add_layer(L* layer, Shape shape = Shape::INVALID);
				void add_layer(const std::string& layer_name, Shape shape = Shape::INVALID);

				const Tensor& forward(const             Tensor& input);
				const Tensor& forward(const std::vector<float>& input);

				const Tensor& backward(const Tensor& error);

				void train(const             Tensor& input, const             Tensor& desired_output);
				void train(const std::vector<float>& input, const std::vector<float>& desired_output);

				void train(const std::vector<            Tensor>& inputs, const std::vector<            Tensor>& desired_outputs, uint64_t epochs = 1, bool verbose = true);
				void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& desired_outputs, uint64_t epochs = 1, bool verbose = true);
				void train(const Dataset::Base& dataset,                                                                          uint64_t epochs = 1, bool verbose = true);

				void test(const             Tensor& input, const             Tensor& desired_output);
				void test(const std::vector<float>& input, const std::vector<float>& desired_output);

				void test(const std::vector<            Tensor>& inputs, const std::vector<            Tensor>& desired_outputs, bool verbose = true);
				void test(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& desired_outputs, bool verbose = true);
				void test(const Dataset::Base& dataset,                                                                          bool verbose = true);

			protected:
				virtual void add_output_layer(void);
				virtual void update_accuracy(const Tensor& output, const Tensor& desired_output, float& accuracy);

			public: // OPERATORS
				template <typename LayerType>
				Base& operator<<(LayerType* layer);

			public: // GETTERS
				Hyperparameters& hyperparameters(void);
				const Shape& input_shape (void) const;
				const Shape& output_shape(void) const;
				float accuracy_training  (void) const;
				float accuracy_testing   (void) const;
			public: // SETTERS
				void input_shape (Shape new_input_shape );
				void output_shape(Shape new_output_shape);
		};

		template <typename L>
		void Base::add_layer(L& layer, Shape specific_shape)
		{
			Shape input_shape = Shape::INVALID;
			Shape       shape = Shape::INVALID;

			if (m_layers.size() == 0 && !layer.is_input())
				add_layer(new Layer::Input());

			if (layer.is_output())
			{
				if (m_output_shape.is_valid())
				{
					if (specific_shape.is_valid()) m_output_shape = specific_shape;
					add_output_layer();
					return;
				}
				else
				{
					std::cerr << "[Base] add_layer: when using Layer::Output/\"output\" set shape of Base before this call"  << std::endl;
					exit(1);
				}
			}

			if (m_layers.size())
				input_shape = (*m_layers.rbegin())->output().shape();
			else if (m_input_shape.is_valid())
				input_shape = m_input_shape;
			else
			{
				std::cerr << "[Base] add_layer: when using Layer::Intput/\"input\" set input_shape of Base before this call or construct/pass with shape"  << std::endl;
				exit(1);
			}

			if (!layer.shape().is_valid())
			{
				if (specific_shape.is_valid())
					shape = specific_shape;
				else
				{
					if (layer.is_input())
					{
						if (m_input_shape.is_valid())
							shape = m_input_shape;
						else
						{
							std::cerr << "[Base] add_layer: when using Layer::Input/\"input\" call this->input_shape(Shape shape) before this call"  << std::endl;
							exit(1);
						}
					}
					else if (layer.is_output())
					{
						if (m_output_shape.is_valid())
							shape = m_output_shape;
						else
						{
							std::cerr << "[Base] add_layer: when using Layer::Output/\"output\" call this->shape(Shape shape) before this call"  << std::endl;
							exit(1);
						}
					}
					else if (!layer.changes_data_shape())
						shape = (*m_layers.rbegin())->output().shape();
					else
					{
						std::cerr << "[Base] add_layer: shape must be specified for `" << layer.name() << "'"  << std::endl;
						exit(1);
					}
				}
			}

			layer.attach_to_neural_network(input_shape, shape, m_hyperparameters);
			// m_layers.push_back(std::make_shared<L>(std::move(layer)));
			m_layers.emplace_back(&layer);
		}

		template <typename L>
		void Base::add_layer(L* layer, Shape shape) { add_layer(*layer, shape); }

		inline void Base::add_output_layer(void) { assert(false && "THIS IS JUST AN INTERFACE"); }
		inline void Base::update_accuracy(const Tensor& output, const Tensor& desired_output, float& accuracy) { (void) output; (void) desired_output; (void) accuracy; assert(false && "THIS IS JUST AN INTERFACE"); }

		// OPERATORS
		template <typename L>
		Base& Base::operator<<(L* layer) { add_layer(*layer); return *this; }

		// GETTERS
		inline       Hyperparameters& Base::hyperparameters  (void)       { return *m_hyperparameters;  }
		inline const Shape&           Base::input_shape      (void) const { return m_input_shape;       }
		inline const Shape&           Base::output_shape     (void) const { return m_output_shape;      }
		inline       float            Base::accuracy_training(void) const { return m_accuracy_training; }
		inline       float            Base::accuracy_testing (void) const { return m_accuracy_testing;  }
		// SETTERS
		inline void Base::input_shape (Shape new_input_shape ) { m_input_shape  = new_input_shape;  }
		inline void Base::output_shape(Shape new_output_shape) { m_output_shape = new_output_shape; }
	}
}

#endif
