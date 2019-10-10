#ifndef _ANNA_CUDA_DEVICE_CUH_
#define _ANNA_CUDA_DEVICE_CUH_

#include <inttypes.h>

namespace Anna
{
	namespace Cuda
	{
		class Device
		{
			private:
				int m_idx;
			public:
				Device(int initial_idx);
				~Device(void);
		};
	}
}

#endif
