#pragma once
#include <cuda_runtime_api.h>
#include <cstdio>

class BaseCudaTestHandler
{
public:

	BaseCudaTestHandler()
	{
	}

	virtual ~BaseCudaTestHandler()
	{
		if (cudaDeviceReset() != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceReset failed!");
		}
	}
};
