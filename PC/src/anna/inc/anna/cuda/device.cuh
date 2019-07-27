#ifndef _ANNA_CUDA_DEVICE_H_
#define _ANNA_CUDA_DEVICE_H_

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
