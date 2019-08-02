#ifndef _ANNA_DATASET_IRIS_H_
#define _ANNA_DATASET_IRIS_H_

#include <inttypes.h>

#include "anna/datasets/base.h"

namespace Anna
{
	namespace Dataset
	{
		class Iris final : public Base
		{
			public:
				Iris(const std::string& dirpath);
				~Iris(void);

			private:
				void load(const std::string& filepath, std::vector<Item>& destination_items) override;
		};
	}
}

#endif
