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
/*
TEST_CASE_METHOD(BaseCudaTestHandler, "check first neighbours list element (neigbours count)", "PlaneSweep")
{
	float x[] = { 1,2,3,4,5,6 };
	float y[] = { 1,2,3,4,5,6 };
	unsigned int type[] = { 0xA, 0xB, 0xC, 0xB, 0xA, 0xC };
	unsigned int ids[] = {1, 1, 1, 2, 2, 2 };
	unsigned int instancesCount = 6;
	unsigned int distanceTreshold = 1;

	GPUUIntKeyProcessor *intKeyProcessor = new GPUUIntKeyProcessor();

	UIntTableGpuHashMap hashMap(6, intKeyProcessor);

	PlaneSweep::Foxtrot::PlaneSweep<float>(x, y, type, ids, instancesCount, distanceTreshold, hashMap);

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
// ----------------------------------------------------------------------------
*/

TEST_CASE_METHOD(BaseCudaTestHandler, "check countNeighbours function", "PlaneSweep")
{
	float hX[] = { 1,2,3,4,5,6 };
	float hY[] = { 1,1,1,1,1,1 };
	UInt hTypes[] = { 0xA, 0xB, 0xC, 0xB, 0xA, 0xC };
	UInt hIds[] = { 1, 1, 1, 2, 2, 2 };
	UInt instancesCount = 6;
	float distanceTreshold = 1.1;
	float distanceTresholdSquared = 1.1 * 1.1;

	// tranfering data from host memory to device memory
	float* cX;
	float* cY;
	UInt* cType;
	UInt* cIds;
	UInt* cResults;

	cudaMalloc(reinterpret_cast<void**>(&cX)		, 6 * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&cY)		, 6 * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&cType)		, 6 * uintSize);
	cudaMalloc(reinterpret_cast<void**>(&cIds)		, 6 * uintSize);
	cudaMalloc(reinterpret_cast<void**>(&cResults)	, 6 * uintSize);

	cudaMemcpy(cX	, hX	, 6 * sizeof(float)	, cudaMemcpyHostToDevice);
	cudaMemcpy(cY	, hY	, 6 * sizeof(float)	, cudaMemcpyHostToDevice);
	cudaMemcpy(cType, hTypes, 6 * uintSize		, cudaMemcpyHostToDevice);
	cudaMemcpy(cIds	, hIds	, 6 * uintSize		, cudaMemcpyHostToDevice);


	dim3 grid;
	int wc = 6;
	findSmallest2D(wc * 32, 256, grid.x, grid.y);

	PlaneSweep::Foxtrot::countNeighbours<<< grid, 256>>> (cX, cY, cType, cIds, instancesCount, distanceTreshold, distanceTresholdSquared, cResults, wc);

	UInt hExpected[] = { 1, 2, 2, 2, 2, 1 };
	UInt hResults[6];

	cudaMemcpy(hResults, cResults, instancesCount, cudaMemcpyDeviceToHost);

	for (int i = 0; i < 6; ++i)
		printf("%d = %u\n", i, hResults[i]);
	
	REQUIRE(std::equal(std::begin(hExpected), std::end(hExpected), hResults));

	cudaFree(cX);
	cudaFree(cY);
	cudaFree(cType);
	cudaFree(cIds);
	cudaFree(cResults);
}
// ----------------------------------------------------------------------------