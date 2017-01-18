#include "..\catch.hpp"

#include "..\..\GPUPatternMining/InstanceTree/InstanceTypedNeighboursMapCreator.h"
#include "../BaseCudaTestHandler.h"
// ----------------------------------------------------------------


TEST_CASE_METHOD(BaseCudaTestHandler, "sizeof unsigned long long is 8 bytes")
{
	REQUIRE(sizeof(unsigned long long) == 8);
}

TEST_CASE_METHOD(BaseCudaTestHandler, "InstanceTypedNeighboursMapCreator | keyCreator host")
{
	using namespace InstanceTypedNeighboursMapCreator;

	FeatureInstance f;
	unsigned short type;

	f.field = 0x000A0001;
	type = 0x000B;

	REQUIRE(createITNMKey(f, type) == 0x0000000A0001000B);
}

/*
Test for graph

         C3
		 |
A1-B1-C1-B2-A2-C2
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "InstanceTypedNeighboursMapCreator | simple 1")
{
	using namespace InstanceTypedNeighboursMapCreator;

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
			, { 0x000A0002 }
			, { 0x000A0002 }
			, { 0x000B0001 }
			, { 0x000B0002 }
			, { 0x000B0002 }
		};

		std::vector<FeatureInstance> hPairsB = {
			{  0x000B0001 }
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
	
	thrust::device_vector<unsigned long long> keys;
	{
		std::vector<unsigned long long> hKeys;

		FeatureInstance fi;
		unsigned short nType;

		fi = { 0x000A0001 };
		nType = 0x000B;
		hKeys.push_back(createITNMKey(fi, nType));

		fi = { 0x000A0002 };
		nType = 0x000B;
		hKeys.push_back(createITNMKey(fi, nType));

		fi = { 0x000A0002 };
		nType = 0x000C;
		hKeys.push_back(createITNMKey(fi, nType));

		fi = { 0x000B0001 };
		nType = 0x000C;
		hKeys.push_back(createITNMKey(fi, nType));

		fi = { 0x000B0002 };
		nType = 0x000C;
		hKeys.push_back(createITNMKey(fi, nType));

		keys = hKeys;
	}

	thrust::device_vector<NeighboursListInfoHolder> values(5);
	
	result->map->getValues(
		thrust::raw_pointer_cast(keys.data())
		, thrust::raw_pointer_cast(values.data())
		, 5
	);
	
	// CHECK

	thrust::device_vector<unsigned int> counts;
	{
		std::vector<unsigned int> hCounts =
		{
			1, 1, 1, 1, 2
		};

		counts = hCounts;
	}

	thrust::device_vector<unsigned int> begins;
	{
		std::vector<unsigned int> hBegins =
		{
			0, 1, 2, 3, 4
		};

		begins = hBegins;
	}
	
	std::vector<NeighboursListInfoHolder> expected
	{
		NeighboursListInfoHolder(1, 0)
		, NeighboursListInfoHolder(1, 1)
		, NeighboursListInfoHolder(1, 2)
		, NeighboursListInfoHolder(1, 3)
		, NeighboursListInfoHolder(2, 4)
	};

	thrust::host_vector<NeighboursListInfoHolder> hValues = values;

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

/*
Test for graph

   A0-C0
	\ /
	B0
	/ \
   D0-C1
	\ /
	 B1
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "InstanceTypedNeighboursMapCreator | complex 1")
{
	using namespace InstanceTypedNeighboursMapCreator;

	// CREATE DATA

	thrust::device_vector<FeatureInstance> pairsA;
	thrust::device_vector<FeatureInstance> pairsB;
	{
		std::vector<FeatureInstance> hPairsA = {
			{ 0xA0000 }
			,{ 0xA0000 }
			,{ 0xB0000 }
			,{ 0xB0000 }
			,{ 0xB0000 }
			,{ 0xB0001 }
			,{ 0xB0001 }
			,{ 0xC0001 }
		};

		std::vector<FeatureInstance> hPairsB = {
			{ 0xB0000 }
			,{ 0xC0000 }
			,{ 0xC0000 }
			,{ 0xC0001 }
			,{ 0xD0000 }
			,{ 0xC0001 }
			,{ 0xD0000 }
			,{ 0xD0000 }
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

	thrust::device_vector<unsigned long long> keys;
	{
		std::vector<unsigned long long> hKeys;

		FeatureInstance fi;
		unsigned short nType;

		fi = { 0xA0000 };
		nType = 0x0000B;
		hKeys.push_back(createITNMKey(fi, nType));

		fi = { 0xA0000 };
		nType = 0x0000C;
		hKeys.push_back(createITNMKey(fi, nType));

		fi = { 0xB0000 };
		nType = 0x0000C;
		hKeys.push_back(createITNMKey(fi, nType));

		fi = { 0xB0000 };
		nType = 0x0000D;
		hKeys.push_back(createITNMKey(fi, nType));

		fi = { 0xB0001 };
		nType = 0x0000C;
		hKeys.push_back(createITNMKey(fi, nType));

		fi = { 0xB0001 };
		nType = 0x0000D;
		hKeys.push_back(createITNMKey(fi, nType));

		fi = { 0xC0001 };
		nType = 0x0000D;
		hKeys.push_back(createITNMKey(fi, nType));

		keys = hKeys;
	}

	thrust::device_vector<NeighboursListInfoHolder> values(7);

	result->map->getValues(
		keys.data().get()
		, values.data().get()
		, 7
	);

	// CHECK

	thrust::device_vector<unsigned int> counts;
	{
		std::vector<unsigned int> hCounts =
		{
			1, 1, 2, 1, 1, 1, 1
		};

		counts = hCounts;
	}

	thrust::device_vector<unsigned int> begins;
	{
		std::vector<unsigned int> hBegins =
		{
			0, 1, 2, 4, 5, 6, 7
		};

		begins = hBegins;
	}

	std::vector<NeighboursListInfoHolder> expected
	{
		NeighboursListInfoHolder(1, 0)
		, NeighboursListInfoHolder(1, 1)
		, NeighboursListInfoHolder(2, 2)
		, NeighboursListInfoHolder(1, 4)
		, NeighboursListInfoHolder(1, 5)
		, NeighboursListInfoHolder(1, 6)
		, NeighboursListInfoHolder(1, 7)
	};

	thrust::host_vector<unsigned int> calculatedCounts = result->counts;
	REQUIRE(std::equal(counts.begin(), counts.end(), calculatedCounts.begin()));

	thrust::host_vector<unsigned int> calculatedBegins = result->begins;
	REQUIRE(std::equal(begins.begin(), begins.end(), calculatedBegins.begin()));
	
	thrust::host_vector<NeighboursListInfoHolder> hValues = values;
	REQUIRE(std::equal(expected.begin(), expected.end(), hValues.begin()));
}
// ------------------------------------------------------------------------------------------