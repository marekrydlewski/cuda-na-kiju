#include "..\catch.hpp"

#include "..\..\GPUPatternMining/InstanceTree/IntanceTablesMapCreator.h"
#include "../BaseCudaTestHandler.h"
// ----------------------------------------------------------------


/*
Test for graph

C3
|
A1-B1-C1-B2-A2-C2
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "InstanceTableMapCreator | simple")
{
	using namespace IntanceTablesMapCreator;

	// CREATE DATA

	thrust::device_vector<FeatureInstance> pairsA;
	thrust::device_vector<FeatureInstance> pairsB;
	{
		/*
		a1 - b1
		a2 - b2
		a2 - c2
		b1 - c1
		b2 - c1
		b2 - c3
		*/
		std::vector<FeatureInstance> hPairsA = {
			{ 0x000A0001 }
			,{ 0x000A0002 }
			,{ 0x000A0002 }
			,{ 0x000B0001 }
			,{ 0x000B0002 }
			,{ 0x000B0002 }
		};

		std::vector<FeatureInstance> hPairsB = {
			{ 0x000B0001 }
			,{ 0x000B0002 }
			,{ 0x000C0002 }
			,{ 0x000C0001 }
			,{ 0x000C0001 }
			,{ 0x000C0003 }
		};

		pairsA = hPairsA;
		pairsB = hPairsB;
	}


	// INVOKE TEST METHOD

	auto result = createTypedNeighboursListMap(
		pairsA
		, pairsB
	);

	// GET DATA FROM MAP

	thrust::device_vector<unsigned int> keys;
	{
		std::vector<unsigned int> hKeys
		{
			0x000A000B
			, 0x000A000C
			, 0x000B000C
		};

		keys = hKeys;
	}

	thrust::device_vector<Entities::InstanceTable> values(3);

	result->map->getValues(
		thrust::raw_pointer_cast(keys.data())
		, thrust::raw_pointer_cast(values.data())
		, 3
	);

	// CHECK

	thrust::device_vector<unsigned int> counts;
	{
		std::vector<unsigned int> hCounts =
		{
			2, 1, 3
		};

		counts = hCounts;
	}

	thrust::device_vector<unsigned int> begins;
	{
		std::vector<unsigned int> hBegins =
		{
			0, 2, 3
		};

		begins = hBegins;
	}

	std::vector<Entities::InstanceTable> expected
	{
		Entities::InstanceTable(2, 0)
		, Entities::InstanceTable(1, 2)
		, Entities::InstanceTable(3, 3)
	};

	thrust::host_vector<Entities::InstanceTable> hValues = values;

	{
		thrust::host_vector<unsigned int> calculatedCounts = result->counts;

		REQUIRE(std::equal(counts.begin(), counts.end(), calculatedCounts.begin()));
	}
	{
		thrust::host_vector<unsigned int> calculatedBegins = result->begins;

		REQUIRE(std::equal(begins.begin(), begins.end(), calculatedBegins.begin()));
	}

	REQUIRE(std::equal(expected.begin(), expected.end(), hValues.begin()));
}