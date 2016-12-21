#include "../catch.hpp"

#define TEST_CUDA_CHECK_RETURN
//--------------------------------------------------------------

#include <vector>

#include "../../GPUPatternMining/Common/MiningCommon.h"

#include "../../GPUPatternMining/PlaneSweep/PlaneSweepFoxtrot.h"

#include "../BaseCudaTestHandler.h"

#include <thrust/device_vector.h>
//--------------------------------------------------------------

using namespace MiningCommon;
//--------------------------------------------------------------


/*
	Test for graph

	A1-B1-C1-B2-A2-C2
*/  
TEST_CASE_METHOD(BaseCudaTestHandler, "Planesweep main 0", "PlaneSweep")
{
	unsigned int instancesCount = 6;
	float distanceTreshold = 1;

	std::vector<float> x = { 1, 2, 3, 4, 5, 6 };
	std::vector<float> y = { 1, 1, 1, 1, 1, 1 };
	
	thrust::device_vector<FeatureInstance> instances(instancesCount);
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		instances[0] = fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xB;
		instances[1] = fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xC;
		instances[2] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xB;
		instances[3] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		instances[4] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xC;
		instances[5] = fi;
	}

	thrust::device_vector<float> dx = x;
	thrust::device_vector<float> dy = y;

	std::shared_ptr<GPUHashMapper<UInt, NeighboursListInfoHolder, GPUKeyProcessor<UInt>>> hashMap;
	thrust::device_vector<FeatureInstance> resultA;
	thrust::device_vector<FeatureInstance> resultB;

	PlaneSweep::Foxtrot::PlaneSweep(
		dx
		, dy
		, instances
		, instancesCount
		, distanceTreshold
		, hashMap
		, resultA
		, resultB
	);

	NeighboursListInfoHolder expectedA0(1, 0);
	NeighboursListInfoHolder expectedA1(2, 1);
	NeighboursListInfoHolder expectedB0(1, 3);
	NeighboursListInfoHolder expectedB1(1, 4);

	std::vector<UInt> resultKeys = { 
		0x000A0000
		, 0x000A0001
		, 0x000B0000
		, 0x000B0001
	};

	thrust::device_vector<UInt> dResultKeys = resultKeys;


	NeighboursListInfoHolder* dResults;
	NeighboursListInfoHolder results[4];

	cudaMalloc(reinterpret_cast<void**>(&dResults), 4 * sizeof(NeighboursListInfoHolder));
	
	hashMap->getValues(
		thrust::raw_pointer_cast(dResultKeys.data())
		, dResults
		, 4);

	cudaMemcpy(results, dResults, 4 * sizeof(NeighboursListInfoHolder), cudaMemcpyDeviceToHost);

	REQUIRE(results[0].count == expectedA0.count);
	REQUIRE(results[0].count == expectedA0.count);

	REQUIRE(results[1].count == expectedA1.count);
	REQUIRE(results[1].count == expectedA1.count);

	REQUIRE(results[2].count == expectedB0.count);
	REQUIRE(results[2].count == expectedB0.count);
	
	REQUIRE(results[3].count == expectedB1.count);
	REQUIRE(results[3].count == expectedB1.count);
}
// ----------------------------------------------------------------------------


/*
	Test for graph

	A0-B0-C0-B1-A1-C1
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "check countNeighbours function", "PlaneSweep")
{
	UInt instancesCount = 6;

	float hX[] = { 1,2,3,4,5,6 };
	float hY[] = { 1,1,1,1,1,1 };


	thrust::device_vector<FeatureInstance> instances(instancesCount);
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		instances[0] = fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xB;
		instances[1] = fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xC;
		instances[2] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xB;
		instances[3] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		instances[4] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xC;
		instances[5] = fi;
	}

	float distanceTreshold = 1.1f;
	float distanceTresholdSquared = 1.1f * 1.1f;

	// tranfering data from host memory to device memory
	float* cX;
	float* cY;
	UInt* cResults;

	cudaMalloc(reinterpret_cast<void**>(&cX)		, 6 * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&cY)		, 6 * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&cResults)	, 6 * uintSize);

	cudaMemcpy(cX			, hX		, 6 * sizeof(float)				, cudaMemcpyHostToDevice);
	cudaMemcpy(cY			, hY		, 6 * sizeof(float)				, cudaMemcpyHostToDevice);

	dim3 grid;
	int warpCount = 6; // same as instances count
	findSmallest2D(warpCount * 32, 256, grid.x, grid.y);

	FeatureInstance* cInstances = thrust::raw_pointer_cast(instances.data());

	PlaneSweep::Foxtrot::countNeighbours<<< grid, 256>>> (
		cX
		, cY
		, cInstances
		, instancesCount
		, distanceTreshold
		, distanceTresholdSquared
		, warpCount
		, cResults
	);

	cudaThreadSynchronize();

	UInt hExpected[] = { 0, 1, 1, 1, 1, 1 };
	UInt hResults[6];

	cudaMemcpy(hResults, cResults, instancesCount * uintSize, cudaMemcpyDeviceToHost);
	
	REQUIRE(std::equal(std::begin(hExpected), std::end(hExpected), hResults));

	cudaFree(cX);
	cudaFree(cY);
	cudaFree(cResults);
}
// ----------------------------------------------------------------------------

/*
Test for graph

A0-B0-C0-B1-A1-C1
*/

TEST_CASE_METHOD(BaseCudaTestHandler, "check findNeighbours function", "PlaneSweep")
{
	// Initialiaze test data

	constexpr UInt instancesCount = 6;

	float hX[] = { 1,2,3,4,5,6 };
	float hY[] = { 1,1,1,1,1,1 };

	thrust::device_vector<FeatureInstance> instances(instancesCount);
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		instances[0] = fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xB;
		instances[1] = fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xC;
		instances[2] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xB;
		instances[3] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		instances[4] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xC;
		instances[5] = fi;
	}

	float distanceTreshold = 1.1f;
	float distanceTresholdSquared = 1.1f * 1.1f;

	UInt hScannedResults[] = { 0, 0, 1, 2, 3, 4 };
	constexpr UInt totalPairs = 5;

	// Tranfering data from host memory to device memory

	float* cX;
	float* cY;
	UInt* cStartPositions;
	FeatureInstance* cResultA;
	FeatureInstance* cResultB;

	constexpr UInt resultTableSize = totalPairs;

	cudaMalloc(reinterpret_cast<void**>(&cX), 6 * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&cY), 6 * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&cStartPositions), 6 * uintSize);
	cudaMalloc(reinterpret_cast<void**>(&cResultA), resultTableSize * sizeof(FeatureInstance));
	cudaMalloc(reinterpret_cast<void**>(&cResultB), resultTableSize * sizeof(FeatureInstance));

	cudaMemcpy(cX, hX, 6 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cY, hY, 6 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cStartPositions, hScannedResults, 6 * uintSize, cudaMemcpyHostToDevice);

	// Setup startup configuration

	dim3 grid;
	int warpCount = 6; 
	findSmallest2D(warpCount * 32, 256, grid.x, grid.y);

	// run tested function

	FeatureInstance* cInstances = thrust::raw_pointer_cast(instances.data()); 

	PlaneSweep::Foxtrot::findNeighbours << < grid, 256 >> > (
		cX
		, cY
		, cInstances
		, instancesCount
		, distanceTreshold
		, distanceTresholdSquared
		, warpCount
		, cStartPositions
		, cResultA
		, cResultB
	);

	cudaThreadSynchronize();

	// Initialize expected output

	FeatureInstance hExpectedA[totalPairs];
	{
		FeatureInstance fi;
		
		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		hExpectedA[0] = fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xB;
		hExpectedA[1] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xB;
		hExpectedA[2] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		hExpectedA[3] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		hExpectedA[4] = fi;
	}

	FeatureInstance hExpectedB[totalPairs];
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xB;
		hExpectedB[0] = fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xC;
		hExpectedB[1] = fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xC;
		hExpectedB[2] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xB;
		hExpectedB[3] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xC;
		hExpectedB[4] = fi;
	}

	// Fetch result from cuda memory

	FeatureInstance hResultA[totalPairs];
	FeatureInstance hResultB[totalPairs];

	cudaMemcpy(hResultA, cResultA, resultTableSize * sizeof(FeatureInstance), cudaMemcpyDeviceToHost);
	cudaMemcpy(hResultB, cResultB, resultTableSize * sizeof(FeatureInstance), cudaMemcpyDeviceToHost);

	// Test output

	REQUIRE(std::equal(std::begin(hExpectedA), std::end(hExpectedA), hResultA));
	REQUIRE(std::equal(std::begin(hExpectedB), std::end(hExpectedB), hResultB));

	// Free allocated resources

	cudaFree(cX);
	cudaFree(cY);
	cudaFree(cStartPositions);
	cudaFree(cResultA);
	cudaFree(cResultB);
}
// ----------------------------------------------------------------------------