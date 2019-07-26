#ifndef _ANNA_LAYER_H_
#define _ANNA_LAYER_H_

#include <inttypes.h>
#include <functional>
#include <map>

#include "anna/layers/all.h"

namespace Anna
{
	namespace Layer
	{
		extern const std::map<const std::string, std::pair<std::function<Layer::Base*(Shape shape)>, bool>> layer_database;

		extern bool changes_data_shape(const std::string& layer_name);
		extern bool is_valid(const std::string& layer_name);
		extern std::function<Layer::Base*(Shape shape)> get_constructor(const std::string& layer_name);
	}
}

#endif
