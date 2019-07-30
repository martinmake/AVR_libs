#ifndef _ANNA_CUDA_STD_CUH_
#define _ANNA_CUDA_STD_CUH_

#include <cuda_runtime.h>

extern cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem=0, cudaStream_t stream=0);

#endif
