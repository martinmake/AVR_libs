#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "anna/datasets/iris.h"

#define  INPUT_COUNT 4
#define OUTPUT_COUNT 3

namespace Anna
{
	namespace Dataset
	{
		Iris::Iris(const std::string& dirpath)
		{
			load(dirpath + "/" + "training.csv",  m_training_items );
			load(dirpath + "/" + "testing.csv",   m_testing_items  );
			load(dirpath + "/" + "unlabeled.csv", m_unlabeled_items);
		}

		Iris::~Iris(void)
		{
		}

		void Iris::load(const std::string& path, std::vector<Item>& destination_items)
		{
			std::ifstream items_csv(path);
			if (!items_csv.is_open())
			{
				std::cout << "[DATASET:IRIS] load: Failed:     " << path << ": No such file" << std::endl;
				return;
			}

			for (std::string line; getline(items_csv, line); )
			{
				std::stringstream line_stream(line);

				destination_items.emplace_back();

				Item& item = *destination_items.rbegin();
				std::vector<float>& input = item.input;
				std::vector<float>& desired_output = item.desired_output;

				for (uint8_t i = 0; i < INPUT_COUNT; i++)
				{
					std::string cell;
					getline(line_stream, cell, ',');
					input.push_back(std::stof(cell));
				}
				std::string cell;
				getline(line_stream, cell);
				if (cell == "setosa")
				{
					desired_output.push_back(0.0);
					desired_output.push_back(0.0);
					desired_output.push_back(1.0);
				}
				else if (cell == "versicolor")
				{
					desired_output.push_back(0.0);
					desired_output.push_back(1.0);
					desired_output.push_back(0.0);
				}
				else if (cell == "virginica")
				{
					desired_output.push_back(1.0);
					desired_output.push_back(0.0);
					desired_output.push_back(0.0);
				}
				else
				{
					std::cout << "ERROR: \"" << cell << "\" IS INVALID SPECIES!)" << std::endl;
					exit(1);
				}
			}

			std::cout << "[DATASET:IRIS] load: Successful: " << path << std::endl;
		}
	}
}
