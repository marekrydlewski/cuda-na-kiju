#include "../catch.hpp"

#define TEST_CUDA_CHECK_RETURN
//--------------------------------------------------------------

#include "../../GPUPatternMining/PlaneSweep/PlaneSweepFoxtrot.cu"

#include "../../GPUPatternMining/MiningCommon.h"

#include "../BaseCudaTestHandler.h"
//--------------------------------------------------------------

/*
	Test for graph

	A1-B1-C1-B2-A2-C2
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "check first neighbours list element (neigbours count)", "PlaneSweep")
{
	float x[] = { 1,2,3,4,5,6 };
	float y[] = { 1,2,3,4,5,6 };
	unsigned int type[] = { 0xA, 0xB, 0xC, 0xB, 0xA, 0xC };
	unsigned int ids[] = {1, 1, 1, 2, 2, 2 };
	unsigned int instancesCount = 6;

	GPUUIntKeyProcessor *intKeyProcessor = new GPUUIntKeyProcessor();

	UIntTableGpuHashMap hashMap(6, intKeyProcessor);

	PlaneSweep::Foxtrot::PlaneSweep<float>(x, y, type, ids, instancesCount, hashMap);

	unsigned int h_resultKeys[] = { 0x000A000B, 0x000A000C, 0x000B000C };
	unsigned int* c_resultKey;

	cudaMalloc(reinterpret_cast<void**>(&c_resultKey), sizeof(3 * uintSize));
	cudaMemcpy(c_resultKey, h_resultKeys, 3 * uintSize, cudaMemcpyHostToDevice);

	unsigned int** c_results;
	unsigned int* d_results[3]; // pointers to GPU memory in host memory
	unsigned int h_result;

	cudaMalloc(reinterpret_cast<void**>(&c_results), 3 * uintPtrSize);
	hashMap.getValues(c_resultKey, c_results, 3);

	cudaMemcpy(d_results, c_results, 3 * uintPtrSize, cudaMemcpyDeviceToHost);

	cudaMemcpy(&h_result, d_results[0], uintSize, cudaMemcpyDeviceToHost);
	REQUIRE(h_result == 2); // |A-B|

	cudaMemcpy(&h_result, d_results[1], uintSize, cudaMemcpyDeviceToHost);
	REQUIRE(h_result == 1); // |A-C|

	cudaMemcpy(&h_result, d_results[3], uintSize, cudaMemcpyDeviceToHost);
	REQUIRE(h_result == 2); // |B-C|
}