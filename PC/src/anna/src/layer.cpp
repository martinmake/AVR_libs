#include <assert.h>
#include <iostream>
#include <functional>
#include <map>

#include "anna/layer.h"

namespace Anna
{
	namespace Layer
	{
		const std::map<const std::string, std::function<Layer::Base*(Shape shape)>> name_to_constructor_mapping =
		{
			{ Input            ::NAME, [](Shape shape) { return new Input            (shape); } },
			{ Output           ::NAME, [](Shape shape) { return new Output           (shape); } },
			{ FullConnected    ::NAME, [](Shape shape) { return new FullConnected    (shape); } },
			{ HyperbolicTangent::NAME, [](Shape shape) { return new HyperbolicTangent(shape); } },
		};

		bool is_valid(const std::string& layer_name)
		{
			return Layer::name_to_constructor_mapping.find(layer_name) != Layer::name_to_constructor_mapping.end();
		}

		Layer::Base& construct(const std::string& layer_name)
		{
			const std::map<const std::string, std::function<Layer::Base*(Shape shape)>>::const_iterator it = Layer::name_to_constructor_mapping.find(layer_name);
			assert(it != Layer::name_to_constructor_mapping.end());

			std::function<Layer::Base*(Shape shape)> constructor = it->second;

			return *constructor(Shape::INVALID);
		}
	}
}
