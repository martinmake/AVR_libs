#ifndef _ANNA_KERNEL_UPDATE_BIASES_H_
#define _ANNA_KERNEL_UPDATE_BIASES_H_

#include <inttypes.h>

namespace Anna
{
	namespace Kernel
	{
		extern __global__ void cuda_update_biases(
					      float* biases,
					const float* error,
					      float  learning_rate,
					      uint64_t neurons_count);
	}
}
#endif
