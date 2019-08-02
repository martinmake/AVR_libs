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
			int device_count;

			cudaCall(cudaGetDeviceCount(&device_count));
			if (device_count)
				printf("[DEVICE COUNT] %d\n", device_count);
			else
				assert(device_count && "[ERROR] NO DEVICES WERE FOUND");

			cudaDeviceProp deviceProp;
			cudaCall(cudaGetDeviceProperties(&deviceProp, m_idx));
			printf("[DEVICE:%d] NAME:                  %s\n",     m_idx, deviceProp.name);
			printf("[DEVICE:%d] MEMORY CLOCK RATE:     %dKHz\n",  m_idx, deviceProp.memoryClockRate);
			printf("[DEVICE:%d] MEMORY BUS WIDTH:      %d\n",     m_idx, deviceProp.memoryBusWidth);
			printf("[DEVICE:%d] MEMORY PEAK BANDWIDTH: %fGB/s\n", m_idx, deviceProp.memoryClockRate * 2.0 * (deviceProp.memoryBusWidth / 8) / 1.0e6);
			cudaCall(cudaSetDevice(m_idx));
		}

		Device::~Device(void)
		{
			cudaCall(cudaDeviceReset());
			cudaCall(cudaProfilerStop());
		}
	}
}
