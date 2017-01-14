#include "..\catch.hpp"
#include "..\BaseCudaTestHandler.h"

#define TEST_CUDA_CHECK_RETURN
//--------------------------------------------------------------

#include "..\..\GPUPatternMining/InstanceTree/InstanceTreeHelpers.h"
#include "../../GPUPatternMining/InstanceTree/IntanceTablesMapCreator.h"
//--------------------------------------------------------------

using namespace InstanceTreeHelpers;
//--------------------------------------------------------------


TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | insert first pair count")
{
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

	thrust::device_vector<Entities::InstanceTable> values;
	{
		std::vector<Entities::InstanceTable> hValues;

		Entities::InstanceTable it;

		it.count = 2;
		it.startIdx = 0;
		hValues.push_back(it);

		it.count = 3;
		it.startIdx = 2;
		hValues.push_back(it);

		it.count = 6;
		it.startIdx = 5;
		hValues.push_back(it);

		values = hValues;
	}
	
	auto proc = GPUUIntKeyProcessor();
	auto map = IntanceTablesMapCreator::InstanceTableMap(5, &proc);


	map.insertKeyValuePairs(
		thrust::raw_pointer_cast(keys.data())
		, thrust::raw_pointer_cast(values.data())
		, 3
	);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	thrust::device_vector<unsigned int> result(3);

	thrust::device_vector<thrust::device_vector<unsigned short>> cliquesData;
	thrust::host_vector<thrust::device_vector<unsigned short>> hcliquesData;
	{
		
		std::vector<unsigned short> first = { 0x000A, 0x000B, 0x000C };
		std::vector<unsigned short> second = { 0x000A, 0x000C, 0x000D };
		std::vector<unsigned short> third = { 0x000B, 0x000C, 0x000D };

		hcliquesData.push_back(first);
		hcliquesData.push_back(second);
		hcliquesData.push_back(third);

		cliquesData = hcliquesData;
	}

	thrust::device_vector<thrust::device_ptr<const unsigned short>> cliques;
	{
		std::vector<thrust::device_ptr<const unsigned short>> hcliques;
		for (const thrust::device_vector<unsigned short>& vec : hcliquesData)
			hcliques.push_back(vec.data());

		cliques = hcliques;
	}

	dim3 insertGrid;
	findSmallest2D(3, 256, insertGrid.x, insertGrid.y);
	
	fillFirstPairCountFromMap <<< insertGrid, 256 >>>(
		map.getBean()
		, thrust::raw_pointer_cast(cliques.data())
		, 3
		, result.data()
	);

	cudaDeviceSynchronize();

	thrust::host_vector<unsigned int> hResult = result;

	std::vector<unsigned int> expected =
	{
		2, 3, 6
	};

	REQUIRE(std::equal(expected.begin(), expected.end(), hResult.begin()));
}
//--------------------------------------------------------------

TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | for groups simple")
{

	thrust::device_vector<unsigned int> counts;
	{
		std::vector<unsigned int> hCount = {
			2, 3, 2, 1
		};

		counts = hCount;
	}

	auto result = forGroups(counts);

	std::vector<unsigned int> expectedGroupNumbers = {
		0, 0, 1, 1, 1, 2, 2, 3
	};

	std::vector<unsigned int> expectedItemNumbers = {
		0, 1, 0, 1, 2, 0, 1, 0
	};

	thrust::host_vector<unsigned int> resultGroupNumbers = result->groupNumbers;
	thrust::host_vector<unsigned int> resultItemNumbers = result->itemNumbers;

	REQUIRE(std::equal(expectedGroupNumbers.begin(), expectedGroupNumbers.end(), resultGroupNumbers.begin()));
	REQUIRE(std::equal(expectedItemNumbers.begin(), expectedItemNumbers.end(), resultItemNumbers.begin()));
	REQUIRE(result->threadCount == 8);
}
//--------------------------------------------------------------


/*
Test for graph

         C3
          |
A1-B1-C1-B2-A2-C2
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | insert first two levels")
{
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

	auto instanceMapResult = IntanceTablesMapCreator::createTypedNeighboursListMap(
		pairsA
		, pairsB
	);

	thrust::device_vector<thrust::device_vector<unsigned short>> cliquesData;
	thrust::host_vector<thrust::device_vector<unsigned short>> hcliquesData;
	{

		std::vector<unsigned short> first = { 0x000A, 0x000B, 0x000C };
		std::vector<unsigned short> second = { 0x000B, 0x000C, 0x000D };

		hcliquesData.push_back(first);
		hcliquesData.push_back(second);

		cliquesData = hcliquesData;
	}

	thrust::device_vector<thrust::device_ptr<const unsigned short>> cliques;
	{
		std::vector<thrust::device_ptr<const unsigned short>> hcliques;
		for (const thrust::device_vector<unsigned short>& vec : hcliquesData)
			hcliques.push_back(vec.data());

		cliques = hcliques;
	}

	thrust::device_vector<unsigned int> groupNumber;
	{
		std::vector<unsigned int> hGroupNumber
		{
			0, 0, 1, 1, 1
		};

		groupNumber = hGroupNumber;
	}

	thrust::device_vector<unsigned int> itemNumber;
	{
		std::vector<unsigned int> hItemNumber
		{
			0, 1, 0, 1, 2
		};

		itemNumber = hItemNumber;
	}

	dim3 insertGrid;
	findSmallest2D(5, 256, insertGrid.x, insertGrid.y);


	thrust::device_vector<FeatureInstance> firstLevel(5);
	thrust::device_vector<FeatureInstance> secondLevel(5);

	writeFirstTwoLevels <<< insertGrid, 256 >>> (
		instanceMapResult->map->getBean()
		, thrust::raw_pointer_cast(cliques.data())
		, groupNumber.data()
		, itemNumber.data()
		, pairsA.data()
		, pairsB.data()
		, 5
		, firstLevel.data()
		, secondLevel.data()
		);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	std::vector<FeatureInstance> expectedFirstLevel;
	{
		FeatureInstance fi;
		
		/*
		a1 - b1
		a2 - b2
		a2 - c2
		b1 - c1
		b2 - c1
		b2 - c3
		*/

		fi.field = 0x000A0001;
		expectedFirstLevel.push_back(fi);

		fi.field = 0x000A0002;
		expectedFirstLevel.push_back(fi);

		fi.field = 0x000B0001;
		expectedFirstLevel.push_back(fi);

		fi.field = 0x000B0002;
		expectedFirstLevel.push_back(fi);

		fi.field = 0x000B0002;
		expectedFirstLevel.push_back(fi);
	}

	std::vector<FeatureInstance> expectedSecondLevel;
	{
		FeatureInstance fi;

		/*
		a1 - b1
		a2 - b2
		a2 - c2
		b1 - c1
		b2 - c1
		b2 - c3
		*/

		fi.field = 0x000B0001;
		expectedSecondLevel.push_back(fi);

		fi.field = 0x000B0002;
		expectedSecondLevel.push_back(fi);

		fi.field = 0x000C0001;
		expectedSecondLevel.push_back(fi);

		fi.field = 0x000C0001;
		expectedSecondLevel.push_back(fi);

		fi.field = 0x000C0003;
		expectedSecondLevel.push_back(fi);
	}
	

	thrust::host_vector<FeatureInstance> resultFirstLevel = firstLevel;
	thrust::host_vector<FeatureInstance> resultSecondLevel = secondLevel;

	REQUIRE(std::equal(expectedFirstLevel.begin(), expectedFirstLevel.end(), resultFirstLevel.begin()));
	REQUIRE(std::equal(expectedSecondLevel.begin(), expectedSecondLevel.end(), resultSecondLevel.begin()));
}
//--------------------------------------------------------------