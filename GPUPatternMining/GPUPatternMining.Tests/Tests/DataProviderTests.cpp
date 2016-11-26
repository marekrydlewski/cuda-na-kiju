#include "../catch.hpp"


#include "../BaseCudaTestHandler.h"
#include "../../GPUPatternMining/SimpleOperations.h"
//--------------------------------------------------------------


TEST_CASE_METHOD(BaseCudaTestHandler, "ComputeArraysSum", "DataProviderTests")
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);

	REQUIRE(cudaStatus == cudaSuccess);
	
	int expected[] = { 11, 22, 33, 44, 55 };

	REQUIRE(std::equal(std::begin(c), std::end(c), std::begin(expected)));
}
