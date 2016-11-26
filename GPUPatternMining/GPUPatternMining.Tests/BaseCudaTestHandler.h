#pragma once
#include <cuda_runtime_api.h>

class BaseCudaTestHandler
{
public:

	BaseCudaTestHandler()
	{
	}

	~BaseCudaTestHandler()
	{
		if (cudaDeviceReset() != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceReset failed!");
		}
	}
};
