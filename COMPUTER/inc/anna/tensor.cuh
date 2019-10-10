#ifndef _ANNA_TENSOR_CUH_
#define _ANNA_TENSOR_CUH_

#include <inttypes.h>

namespace Anna
{
	namespace Cuda
	{
		extern void substract(float* lhs, const float* rhs, uint64_t count);
	}
}

#endif
