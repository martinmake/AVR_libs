#include <assert.h>
#include <iostream>
#include <functional>
#include <map>

#include "layer.h"

namespace Anna
{
	namespace Layer
	{
		const std::map<const std::string, Layer::Data> datamap =
		{
			{ Input            ::NAME, { [](Shape shape) { return new Input            (shape); }, Input            ::CHANGES_DATA_SHAPE, Input            ::IS_OUTPUT } },
			{ Output           ::NAME, { [](Shape shape) { return new Output           (shape); }, Output           ::CHANGES_DATA_SHAPE, Output           ::IS_OUTPUT } },
			{ FullConnected    ::NAME, { [](Shape shape) { return new FullConnected    (shape); }, FullConnected    ::CHANGES_DATA_SHAPE, FullConnected    ::IS_OUTPUT } },
			{ HyperbolicTangent::NAME, { [](Shape shape) { return new HyperbolicTangent(shape); }, HyperbolicTangent::CHANGES_DATA_SHAPE, HyperbolicTangent::IS_OUTPUT } },
		};

		bool is_valid(const std::string& layer_name)
		{
			const auto& it = Layer::datamap.find(layer_name);
			return it != Layer::datamap.end();
		}

		const Layer::Data& data(const std::string& layer_name)
		{
			const std::map<const std::string, Layer::Data>::const_iterator it = Layer::datamap.find(layer_name);
			assert(it != Layer::datamap.end());

			return it->second;
		}
	}
}
