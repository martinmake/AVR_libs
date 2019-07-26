#ifndef _ANNA_LAYER_H_
#define _ANNA_LAYER_H_

#include "anna/layers/all.h"

namespace Anna
{
	namespace Layer
	{
		struct Data
		{
			const std::function<Layer::Base*(Shape shape)> constructor;
			const bool changes_data_shape;
			const bool is_output;
		};

		extern bool is_valid(const std::string& layer_name);
		extern const Layer::Data& data(const std::string& layer_name);
	}
}

#endif
