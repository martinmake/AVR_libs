#ifndef _ANNA_LAYER_H_
#define _ANNA_LAYER_H_

#include "anna/layers/all.h"

namespace Anna
{
	namespace Layer
	{
		extern bool is_valid(const std::string& layer_name);
		extern Layer::Base& construct(const std::string& layer_name);
	}
}

#endif
