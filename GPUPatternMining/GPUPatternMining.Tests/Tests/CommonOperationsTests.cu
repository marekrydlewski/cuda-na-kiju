#include "../catch.hpp"

#define TEST_CUDA_CHECK_RETURN
//--------------------------------------------------------------

#include "../BaseCudaTestHandler.h"

#include "../../GPUPatternMining/Common/MiningCommon.h"
#include "../../GPUPatternMining/Common/CommonOperations.h"

#include "../../GPUPatternMining.Contract/Enity/FeatureInstance.h"
//--------------------------------------------------------------


/*
	The reason for next five following tests is to check if the fields in the FeatureInstance inner struct
	are placed in the right order
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Test featureInstance comparision (a < b) featureId", "Sanity")
{
	FeatureInstance a;
	a.fields.featureId = 0xA;
	a.fields.instanceId = 0x0;

	FeatureInstance b;
	b.fields.featureId = 0xB;
	b.fields.instanceId = 0x0;

	REQUIRE(a.field < b.field);
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Test featureInstance comparision (a < b) instanceId", "Sanity")
{
	FeatureInstance a;
	a.fields.featureId = 0xA;
	a.fields.instanceId = 0x0;

	FeatureInstance b;
	b.fields.featureId = 0xA;
	b.fields.instanceId = 0x1;

	REQUIRE(a.field < b.field);
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Test featureInstance comparision (a == b)", "Sanity")
{
	FeatureInstance a;
	a.fields.featureId = 0xB;
	a.fields.instanceId = 0x99;

	FeatureInstance b;
	b.fields.featureId = 0xB;
	b.fields.instanceId = 0x99;

	REQUIRE(a.field == b.field);
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Test featureInstance comparision (a > b) featureId", "Sanity")
{
	FeatureInstance a;
	a.fields.featureId = 0xB;
	a.fields.instanceId = 0x0;

	FeatureInstance b;
	b.fields.featureId = 0xA;
	b.fields.instanceId = 0x0;

	REQUIRE(a.field > b.field);
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Test featureInstance comparision (a > b) instanceId", "Sanity")
{
	FeatureInstance a;
	a.fields.featureId = 0xB;
	a.fields.instanceId = 0x2;

	FeatureInstance b;
	b.fields.featureId = 0xB;
	b.fields.instanceId = 0x1;

	REQUIRE(a.field > b.field);
}


TEST_CASE_METHOD(BaseCudaTestHandler, "Reduce key operation", "Common operations")
{
	constexpr UInt totalPairs = 5;

	thrust::device_vector<FeatureInstance> hExpectedA(totalPairs);
	{
		FeatureInstance fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xA;
		hExpectedA[0] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		hExpectedA[1] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xA;
		hExpectedA[2] = fi;

		fi.fields.instanceId = 0x0;
		fi.fields.featureId = 0xB;
		hExpectedA[3] = fi;

		fi.fields.instanceId = 0x1;
		fi.fields.featureId = 0xB;
		hExpectedA[4] = fi;
	}

	thrust::device_vector<FeatureInstance> uniques(totalPairs);
	thrust::device_vector<UInt> indices(totalPairs);
	thrust::device_vector<UInt> counts(totalPairs);

	UInt entryCount = thrust::reduce_by_key(
		hExpectedA.begin(),
		hExpectedA.end(),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				thrust::counting_iterator<UInt>(0),
				thrust::constant_iterator<UInt>(1)
			)
		),
		uniques.begin(),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				indices.begin(),
				counts.end()
			)
		),
		MiningCommon::InstanceEquality<FeatureInstance>(),
		MiningCommon::FirstIndexAndCount<UInt>()
	).first - uniques.begin();

	REQUIRE(entryCount == 4);
}

__global__ void runScan(int* data)
{

	MiningCommon::intraWarpScan<int>(thrust::raw_pointer_cast(data));
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Scan test", "Common operations")
{
	std::vector<int> numbers = {
		1, 1, 1, 1, 1, 
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1
	};

	thrust::device_vector<int> data = numbers;

	runScan<<<1, 32>>>(thrust::raw_pointer_cast(data.data()));

	thrust::host_vector<int> result = data;

	std::vector<int> expected = { 
		0 , 1 , 2 , 3 , 4 ,
		5 , 6 , 7 , 8 , 9 ,
		10, 11, 12, 13, 14,
		15, 16, 17, 18, 19,
		20, 21, 22, 23, 24,
		25, 26, 27, 28, 29,
		30, 31
	};

	REQUIRE(std::equal(expected.begin(), expected.end(), result.begin()));

}