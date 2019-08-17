#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "anna/datasets/mnist.h"

#define  INPUT_COUNT (28 * 28)
#define OUTPUT_COUNT 10

namespace Anna
{
	namespace Dataset
	{
		Mnist::Mnist(const std::string& dirpath)
		{
			load(dirpath + "/" + "training",  m_training_items);
			load(dirpath + "/" + "testing",   m_testing_items);
			load(dirpath + "/" + "unlabeled", m_unlabeled_items);
		}

		Mnist::~Mnist(void)
		{
		}

		void Mnist::load(const std::string& path, std::vector<Item>& destination_items)
		{
			const std::string images_path = path + "/" + "images.gz";
			const std::string labels_path = path + "/" + "labels.gz";
			std::ifstream images(images_path, std::ios::binary);
			std::ifstream labels(labels_path, std::ios::binary);
			if (!images.is_open())
			{
				std::cout << "[DATASET:MNIST] load: Failed:     " << images_path << ": No such file" << std::endl;
				return;
			}
			if (!labels.is_open())
			{
				std::cout << "[DATASET:MNIST] load: Failed:     " << labels_path << ": No such file" << std::endl;
				return;
			}

			int32_t count = 0;

			labels.seekg(4);
			labels.read((char*) &count + 3, 1);
			labels.read((char*) &count + 2, 1);
			labels.read((char*) &count + 1, 1);
			labels.read((char*) &count + 0, 1);
			images.seekg(16);

			for (int32_t i = 0; i < count; i++)
			{
				destination_items.emplace_back();

				Item& item = *destination_items.rbegin();
				std::vector<float>& input = item.input;
				std::vector<float>& desired_output = item.desired_output;

				for (uint16_t i = 0; i < INPUT_COUNT; i++)
				{
					uint8_t pixel = 0;
					images.read((char*) &pixel, 1);
					input.push_back(pixel ? 1.0 : 0.0);
				}

				uint8_t label = 0;
				labels.read((char*) &label, 1);
				for (uint8_t i = 0; i <= 9; i++)
					desired_output.push_back(i == label ? 1.0 : 0.0);
			}

			std::cout << "[DATASET:MNIST] load: Successful: " << path << std::endl;
		}
	}
}
