#ifndef _ANNA_CUDA_DEBUG_H_
#define _ANNA_CUDA_DEBUG_H_

#define cudaCall(call)                                                                                 \
{                                                                                                      \
	const cudaError_t error = call;                                                                \
	if (error != cudaSuccess)                                                                      \
	{                                                                                              \
		printf("[ERROR:%d]: %s:%d, %s", error, __FILE__, __LINE__, cudaGetErrorString(error)); \
		exit(-10 * error);                                                                     \
	}                                                                                              \
}

#endif
