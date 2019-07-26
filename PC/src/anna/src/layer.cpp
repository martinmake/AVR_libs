#include <assert.h>
#include <iostream>

#include "layer.h"

namespace Anna
{
	namespace Layer
	{
		const std::map<const std::string, std::pair<std::function<Layer::Base*(Shape shape)>, bool>> layer_database =
		{
			{ ANNA_LAYER_INPUT_NAME,              { [](Shape shape) { return new Layer::Input            (shape); }, ANNA_LAYER_INPUT_CHANGES_DATA_SHAPE              } },
			{ ANNA_LAYER_OUTPUT_NAME,             { [](Shape shape) { return new Layer::Output           (shape); }, ANNA_LAYER_OUTPUT_CHANGES_DATA_SHAPE             } },
			{ ANNA_LAYER_FULL_CONNECTED_NAME,     { [](Shape shape) { return new Layer::FullConnected    (shape); }, ANNA_LAYER_FULL_CONNECTED_CHANGES_DATA_SHAPE     } },
			{ ANNA_LAYER_HYPERBOLIC_TANGENT_NAME, { [](Shape shape) { return new Layer::HyperbolicTangent(shape); }, ANNA_LAYER_HYPERBOLIC_TANGENT_CHANGES_DATA_SHAPE } },
		};

		bool changes_data_shape(const std::string& layer_name)
		{
			const std::map<const std::string, std::pair<std::function<Layer::Base*(Shape shape)>, bool>>::const_iterator it = Layer::layer_database.find(layer_name);
			assert(it != Layer::layer_database.end());

			return it->second.second;
		}

		bool is_valid(const std::string& layer_name)
		{
			const auto& it = Layer::layer_database.find(layer_name);
			return it != Layer::layer_database.end();
		}

		std::function<Layer::Base*(Shape shape)> get_constructor(const std::string& layer_name)
		{
			const std::map<const std::string, std::pair<std::function<Layer::Base*(Shape shape)>, bool>>::const_iterator it = Layer::layer_database.find(layer_name);
			assert(it != layer_database.end());

			return it->second.first;
		}
	}
}
