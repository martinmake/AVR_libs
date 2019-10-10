#ifndef _ANNA_DATASET_MNIST_H_
#define _ANNA_DATASET_MNIST_H_

#include <inttypes.h>

#include "anna/datasets/base.h"

namespace Anna
{
	namespace Dataset
	{
		class Mnist final : public Base
		{
			public:
				Mnist(const std::string& dirpath);
				~Mnist(void);

			private:
				void load(const std::string& path, std::vector<Item>& destination_items) override;
		};
	}
}

#endif
