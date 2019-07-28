#ifndef _ANNA_CUDA_DEBUG_H_
#define _ANNA_CUDA_DEBUG_H_

#include <stdio.h>
#include <cuda_runtime.h>

#define cudaCall(call)                                                                                 \
{                                                                                                      \
	const cudaError_t error = call;                                                                \
	if (error != cudaSuccess)                                                                      \
	{                                                                                              \
		printf("[ERROR:%d]: %s:%d, %s\n", error, __FILE__, __LINE__, cudaGetErrorString(error)); \
		exit(-10 * error);                                                                     \
	}                                                                                              \
}

#endif
