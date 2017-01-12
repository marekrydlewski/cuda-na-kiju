#include "../catch.hpp"

#define TEST_CUDA_CHECK_RETURN
//--------------------------------------------------------------

#include <vector>

#include "../../GPUPatternMining/Common/MiningCommon.h"

#include "../../GPUPatternMining/PlaneSweep/InstanceTablePlaneSweep.h"

#include "../BaseCudaTestHandler.h"

#include <thrust/device_vector.h>
#include "../../GPUPatternMining/Entities/InstanceTable.h"
//--------------------------------------------------------------

using namespace MiningCommon;
//--------------------------------------------------------------


/*
	Test for graph

	A1-B1-C1-B2-A2-C2
*/ 
TEST_CASE_METHOD(BaseCudaTestHandler,"PlaneSweep_instanceTable | Planesweep main")
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

	PlaneSweepTableInstanceResultPtr result = std::make_shared<PlaneSweepTableInstanceResult>();

	PlaneSweep::InstanceTable::PlaneSweep(
		dx
		, dy
		, instances
		, instancesCount
		, distanceTreshold
		, result
	);

	cudaDeviceSynchronize();

	{
		thrust::host_vector<unsigned int> indicies = result->indices;
		printf("indices\n");

		for (unsigned int val : indicies)
			printf("%u\n", val);
	}

	Entities::InstanceTable expectedAB(2, 0);
	Entities::InstanceTable expectedAC(1, 2);
	Entities::InstanceTable expectedBC(2, 3);

	std::vector<UInt> resultKeys = {
		0x000A000B
		, 0x000A000C
		, 0x000B000C
	};

	thrust::device_vector<UInt> dResultKeys = resultKeys;


	Entities::InstanceTable* dResults;
	Entities::InstanceTable results[3];

	cudaMalloc(reinterpret_cast<void**>(&dResults), 3 * sizeof(Entities::InstanceTable));
	
	result->instanceTableMap->getValues(
		thrust::raw_pointer_cast(dResultKeys.data())
		, dResults
		, 3);

	cudaMemcpy(results, dResults, 3 * sizeof(Entities::InstanceTable), cudaMemcpyDeviceToHost);

	REQUIRE(results[0].count == expectedAB.count);
	REQUIRE(results[0].startIdx == expectedAB.startIdx);

	REQUIRE(results[1].count == expectedAC.count);
	REQUIRE(results[1].startIdx == expectedAC.startIdx);

	REQUIRE(results[2].count == expectedBC.count);
	REQUIRE(results[2].startIdx == expectedBC.startIdx);
}
// ----------------------------------------------------------------------------

TEST_CASE_METHOD(BaseCudaTestHandler,"PlaneSweep_instanceTable | Planesweep main 1 (Far)")
{
	unsigned int instancesCount = 64;
	float distanceTreshold = 64;

	std::vector<float> x(instancesCount);
	std::vector<float> y(instancesCount);
	
	for (int i = 1; i < 63; ++i)
	{
		x[i] = i;
		y[i] = i * 100;
	}

	x[0] = 0;
	y[0] = 1;

	x[63] = 63;
	y[63] = 1;


	std::vector<FeatureInstance> hostInstances(instancesCount);
	
	for (int i = 1; i < 63; ++i)
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0 + i;
		fi.fields.featureId = 0xB;
		hostInstances[i] = fi;
	}

	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		hostInstances[0] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xC;
		hostInstances[63] = fi;
	}


	thrust::device_vector<FeatureInstance> instances = hostInstances;
	thrust::device_vector<float> dx = x;
	thrust::device_vector<float> dy = y;


	PlaneSweepTableInstanceResultPtr result = std::make_shared<PlaneSweepTableInstanceResult>();

	PlaneSweep::InstanceTable::PlaneSweep(
		dx
		, dy
		, instances
		, instancesCount
		, distanceTreshold
		, result
	);

	cudaDeviceSynchronize();

	Entities::InstanceTable expectedAC(1, 0);
	
	std::vector<UInt> resultKeys = {
		0x000A000C
	};

	thrust::device_vector<UInt> dResultKeys = resultKeys;

	Entities::InstanceTable* dResults;
	Entities::InstanceTable results[1];

	cudaMalloc(reinterpret_cast<void**>(&dResults), 1 * sizeof(Entities::InstanceTable));

	result->instanceTableMap->getValues(
		thrust::raw_pointer_cast(dResultKeys.data())
		, dResults
		, 1);

	cudaMemcpy(results, dResults, 1 * sizeof(Entities::InstanceTable), cudaMemcpyDeviceToHost);

	REQUIRE(results[0].count == expectedAC.count);
	REQUIRE(results[0].startIdx == expectedAC.startIdx);
}
// ----------------------------------------------------------------------------

/*
	Test for graph

	A0-B0-C0-B1-A1-C1
*/ 
TEST_CASE_METHOD(BaseCudaTestHandler,"PlaneSweep_instanceTable | check countNeighbours function")
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

	PlaneSweep::InstanceTable::countNeighbours<<< grid, 256>>> (
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


TEST_CASE_METHOD(BaseCudaTestHandler,"PlaneSweep_instanceTable | check countNeighbours function (far)")
{
	unsigned int instancesCount = 64;
	float distanceTreshold = 64;

	std::vector<float> x(instancesCount);
	std::vector<float> y(instancesCount);

	for (int i = 1; i < 63; ++i)
	{
		x[i] = i;
		y[i] = i * 100;
	}

	x[0] = 0;
	y[0] = 1;

	x[63] = 63;
	y[63] = 1;


	std::vector<FeatureInstance> hostInstances(instancesCount);

	for (int i = 1; i < 63; ++i)
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0 + i;
		fi.fields.featureId = 0xB;
		hostInstances[i] = fi;
	}

	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		hostInstances[0] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		hostInstances[63] = fi;
	}


	thrust::device_vector<FeatureInstance> instances = hostInstances;
	thrust::device_vector<float> dx = x;
	thrust::device_vector<float> dy = y;

	thrust::device_vector<UInt> result(instancesCount);

	dim3 grid;
	dim3 block(256, 1, 1);
	int warpCount = instancesCount; // same as instances count
	findSmallest2D(warpCount * 32, 256, grid.x, grid.y);

	FeatureInstance* cInstances = thrust::raw_pointer_cast(instances.data());

	float  distanceTresholdSquared = distanceTreshold * distanceTreshold;

	PlaneSweep::InstanceTable::countNeighbours <<< grid, block >>> (
		thrust::raw_pointer_cast(dx.data())
		, thrust::raw_pointer_cast(dy.data())
		, cInstances
		, instancesCount
		, distanceTreshold
		, distanceTresholdSquared
		, warpCount
		, thrust::raw_pointer_cast(result.data())
		);

	cudaThreadSynchronize();
	
	std::vector<UInt> expected(instancesCount);

	for (int i = 0; i < 63; ++i)
		expected[i] = 0;

	expected[63] = 1;

	thrust::host_vector<UInt> hResult = result;

	REQUIRE(std::equal(hResult.begin(), hResult.end(), expected.begin()));
	
	dx.clear();
	dy.clear();
	result.clear();
}
// ----------------------------------------------------------------------------


TEST_CASE_METHOD(BaseCudaTestHandler,"PlaneSweep_instanceTable | check countNeighbours function (one per warp iteration)")
{
	unsigned int instancesCount = 64;
	float distanceTreshold = 64;

	std::vector<float> x(instancesCount);
	std::vector<float> y(instancesCount);

	for (int i = 1; i < 63; ++i)
	{
		x[i] = i;
		y[i] = i * 100;
	}

	x[0] = 0;
	y[0] = 1;

	x[32] = 32;
	y[32] = 1;

	x[63] = 63;
	y[63] = 1;


	std::vector<FeatureInstance> hostInstances(instancesCount);

	for (int i = 1; i < 63; ++i)
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0 + i;
		fi.fields.featureId = 0xB;
		hostInstances[i] = fi;
	}

	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		hostInstances[0] = fi;

		fi.fields.instanceId = 0x2;
		fi.fields.featureId = 0xA;
		hostInstances[32] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		hostInstances[63] = fi;
	}


	thrust::device_vector<FeatureInstance> instances = hostInstances;
	thrust::device_vector<float> dx = x;
	thrust::device_vector<float> dy = y;

	thrust::device_vector<UInt> result(instancesCount);

	dim3 grid;
	dim3 block(256, 1, 1);
	int warpCount = instancesCount; // same as instances count
	findSmallest2D(warpCount * 32, 256, grid.x, grid.y);

	FeatureInstance* cInstances = thrust::raw_pointer_cast(instances.data());

	float  distanceTresholdSquared = distanceTreshold * distanceTreshold;

	PlaneSweep::InstanceTable::countNeighbours <<< grid, block >>> (
		thrust::raw_pointer_cast(dx.data())
		, thrust::raw_pointer_cast(dy.data())
		, cInstances
		, instancesCount
		, distanceTreshold
		, distanceTresholdSquared
		, warpCount
		, thrust::raw_pointer_cast(result.data())
		);

	cudaThreadSynchronize();

	std::vector<UInt> expected(instancesCount);

	for (int i = 0; i < 63; ++i)
		expected[i] = 0;

	expected[63] = 2;
	expected[32] = 1;

	thrust::host_vector<UInt> hResult = result;

	REQUIRE(std::equal(hResult.begin(), hResult.end(), expected.begin()));

	dx.clear();
	dy.clear();
	result.clear();
}
// ----------------------------------------------------------------------------


/*
Test for graph

A0-B0-C0-B1-A1-C1
*/
TEST_CASE_METHOD(BaseCudaTestHandler,"PlaneSweep_instanceTable | check findNeighbours function")
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

	PlaneSweep::InstanceTable::findNeighbours << < grid, 256 >> > (
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


TEST_CASE_METHOD(BaseCudaTestHandler,"PlaneSweep_instanceTable | check findNeighbours function (far)")
{
	unsigned int instancesCount = 64;
	float distanceTreshold = 64;

	std::vector<float> x(instancesCount);
	std::vector<float> y(instancesCount);

	for (int i = 1; i < 63; ++i)
	{
		x[i] = i;
		y[i] = i * 100;
	}

	x[0] = 0;
	y[0] = 1;

	x[63] = 63;
	y[63] = 1;


	std::vector<FeatureInstance> hostInstances(instancesCount);

	for (int i = 1; i < 63; ++i)
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0 + i;
		fi.fields.featureId = 0xB;
		hostInstances[i] = fi;
	}

	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		hostInstances[0] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		hostInstances[63] = fi;
	}


	thrust::device_vector<FeatureInstance> instances = hostInstances;
	thrust::device_vector<float> dx = x;
	thrust::device_vector<float> dy = y;

	thrust::device_vector<UInt> result(instancesCount);

	dim3 grid;
	dim3 block(256, 1, 1);
	int warpCount = instancesCount; // same as instances count
	findSmallest2D(warpCount * 32, 256, grid.x, grid.y);

	FeatureInstance* cInstances = thrust::raw_pointer_cast(instances.data());

	float  distanceTresholdSquared = distanceTreshold * distanceTreshold;

	std::vector<UInt> startPositions(64, 0);
	thrust::device_vector<UInt> dStartPositions = startPositions;

	thrust::device_vector<FeatureInstance> dResultA(1);
	thrust::device_vector<FeatureInstance> dResultB(1);

	PlaneSweep::InstanceTable::findNeighbours <<< grid, block >>> (
		thrust::raw_pointer_cast(dx.data())
		, thrust::raw_pointer_cast(dy.data())
		, cInstances
		, instancesCount
		, distanceTreshold
		, distanceTresholdSquared
		, warpCount
		, thrust::raw_pointer_cast(dStartPositions.data())
		, thrust::raw_pointer_cast(dResultA.data())
		, thrust::raw_pointer_cast(dResultB.data())
		);

	cudaThreadSynchronize();

	// Initialize expected output

	FeatureInstance hExpectedA[1];
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		hExpectedA[0] = fi;
	}

	FeatureInstance hExpectedB[1];
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		hExpectedB[0] = fi;
	}

	// Fetch result from cuda memory

	thrust::host_vector<FeatureInstance> resultsA = dResultA;
	thrust::host_vector<FeatureInstance> resultsB = dResultB;

	// Test output

	REQUIRE(std::equal(std::begin(hExpectedA), std::end(hExpectedA), resultsA.begin()));
	REQUIRE(std::equal(std::begin(hExpectedB), std::end(hExpectedB), resultsB.begin()));
}
// ----------------------------------------------------------------------------

TEST_CASE_METHOD(BaseCudaTestHandler,"PlaneSweep_instanceTable | check findNeighbours function (one per warp iteration)")
{
	unsigned int instancesCount = 64;
	float distanceTreshold = 64;

	std::vector<float> x(instancesCount);
	std::vector<float> y(instancesCount);

	for (int i = 1; i < 63; ++i)
	{
		x[i] = i;
		y[i] = i * 100;
	}

	x[0] = 0;
	y[0] = 1;

	x[32] = 32;
	y[32] = 1;

	x[63] = 63;
	y[63] = 1;


	std::vector<FeatureInstance> hostInstances(instancesCount);

	for (int i = 1; i < 63; ++i)
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0 + i;
		fi.fields.featureId = 0xB;
		hostInstances[i] = fi;
	}

	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		hostInstances[0] = fi;

		fi.fields.instanceId = 0x2;
		fi.fields.featureId = 0xA;
		hostInstances[32] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		hostInstances[63] = fi;
	}


	thrust::device_vector<FeatureInstance> instances = hostInstances;
	thrust::device_vector<float> dx = x;
	thrust::device_vector<float> dy = y;

	thrust::device_vector<UInt> result(instancesCount);

	dim3 grid;
	dim3 block(256, 1, 1);
	int warpCount = instancesCount; // same as instances count
	findSmallest2D(warpCount * 32, 256, grid.x, grid.y);

	FeatureInstance* cInstances = thrust::raw_pointer_cast(instances.data());

	float  distanceTresholdSquared = distanceTreshold * distanceTreshold;

	std::vector<UInt> startPositions(64, 0);
	{
		startPositions[63] = 1;
	}
	thrust::device_vector<UInt> dStartPositions = startPositions;


	thrust::device_vector<FeatureInstance> dResultA(3);
	thrust::device_vector<FeatureInstance> dResultB(3);

	PlaneSweep::InstanceTable::findNeighbours << < grid, block >> > (
		thrust::raw_pointer_cast(dx.data())
		, thrust::raw_pointer_cast(dy.data())
		, cInstances
		, instancesCount
		, distanceTreshold
		, distanceTresholdSquared
		, warpCount
		, thrust::raw_pointer_cast(dStartPositions.data())
		, thrust::raw_pointer_cast(dResultA.data())
		, thrust::raw_pointer_cast(dResultB.data())
		);

	cudaThreadSynchronize();

	// Initialize expected output

	FeatureInstance hExpectedA[3];
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		hExpectedA[0] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		hExpectedA[1] = fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		hExpectedA[2] = fi;
	}

	FeatureInstance hExpectedB[3];
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x2;
		fi.fields.featureId = 0xA;
		hExpectedB[0] = fi;

		fi.fields.instanceId = 0x2;
		fi.fields.featureId = 0xA;
		hExpectedB[1] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		hExpectedB[2] = fi;
	}

	// Fetch result from cuda memory

	thrust::host_vector<FeatureInstance> resultsA = dResultA;
	thrust::host_vector<FeatureInstance> resultsB = dResultB;

	// Test output
	
	//for (int i = 0; i < 3; ++i)
	//{
	//	printf("exp [%#08x, %#08x] res [%#08x, %#08x] \n"
	//		, hExpectedA[i].field, hExpectedB[i].field
	//		, resultsA[i].field  , resultsB[i].field
	//	);
	//}
	

	REQUIRE(std::equal(std::begin(hExpectedA), std::end(hExpectedA), resultsA.begin()));
	REQUIRE(std::equal(std::begin(hExpectedB), std::end(hExpectedB), resultsB.begin()));
}
// ----------------------------------------------------------------------------