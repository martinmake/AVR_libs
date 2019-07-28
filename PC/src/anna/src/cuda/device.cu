#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "cuda/device.cuh"
#include "cuda/debug.cuh"

namespace Anna
{
	namespace Cuda
	{
		Device::Device(int initial_idx)
			: m_idx(initial_idx)
		{
			cudaDeviceProp m_deviceProp;
			cudaCall(cudaGetDeviceProperties(&m_deviceProp, m_idx));
			printf("[DEVICE:%d] %s\n", m_idx, m_deviceProp.name);
			cudaCall(cudaSetDevice(m_idx));
		}

		Device::~Device(void)
		{
		// cudaCall(cudaDeviceReset());
		// cudaCall(cudaProfilerStop());
		}
	}
}
