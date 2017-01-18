#include "..\catch.hpp"
#include "..\BaseCudaTestHandler.h"

#define TEST_CUDA_CHECK_RETURN
//---------------------------------------------------------------------------------------------------

#include "..\..\GPUPatternMining/InstanceTree/InstanceTreeHelpers.h"
#include "../../GPUPatternMining/InstanceTree/IntanceTablesMapCreator.h"
//---------------------------------------------------------------------------------------------------

using namespace InstanceTreeHelpers;
//---------------------------------------------------------------------------------------------------

typedef thrust::device_vector<unsigned int> UIntThrustVector;
typedef std::shared_ptr<UIntThrustVector> UIntThrustVectorPtr;

typedef thrust::device_vector<FeatureInstance> FeatureInstanceThrustVector;
typedef std::shared_ptr<FeatureInstanceThrustVector> FeatureInstanceThrustVectorPtr;
//---------------------------------------------------------------------------------------------------

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

TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | for groups last zero")
{

	thrust::device_vector<unsigned int> counts;
	{
		std::vector<unsigned int> hCount = {
			2, 3, 2, 0
		};

		counts = hCount;
	}

	auto result = forGroups(counts);

	std::vector<unsigned int> expectedGroupNumbers = {
		0, 0, 1, 1, 1, 2, 2
	};

	std::vector<unsigned int> expectedItemNumbers = {
		0, 1, 0, 1, 2, 0, 1
	};

	thrust::host_vector<unsigned int> resultGroupNumbers = result->groupNumbers;
	thrust::host_vector<unsigned int> resultItemNumbers = result->itemNumbers;

	REQUIRE(std::equal(expectedGroupNumbers.begin(), expectedGroupNumbers.end(), resultGroupNumbers.begin()));
	REQUIRE(std::equal(expectedItemNumbers.begin(), expectedItemNumbers.end(), resultItemNumbers.begin()));
	REQUIRE(result->threadCount == 7);
}
//--------------------------------------------------------------

TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | for groups inner zeros")
{

	thrust::device_vector<unsigned int> counts;
	{
		std::vector<unsigned int> hCount = {
			1, 2, 0, 0, 1
		};

		counts = hCount;
	}

	auto result = forGroups(counts);

	std::vector<unsigned int> expectedGroupNumbers = {
		0, 1, 1, 4
	};

	std::vector<unsigned int> expectedItemNumbers = {
		0, 0, 1, 0
	};

	thrust::host_vector<unsigned int> resultGroupNumbers = result->groupNumbers;
	thrust::host_vector<unsigned int> resultItemNumbers = result->itemNumbers;

	REQUIRE(std::equal(expectedGroupNumbers.begin(), expectedGroupNumbers.end(), resultGroupNumbers.begin()));
	REQUIRE(std::equal(expectedItemNumbers.begin(), expectedItemNumbers.end(), resultItemNumbers.begin()));
	REQUIRE(result->threadCount == 4);
}
//--------------------------------------------------------------

TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | for groups inner zeros extended")
{

	thrust::device_vector<unsigned int> counts;
	{
		std::vector<unsigned int> hCount = {
			1, 2, 0, 0, 4, 1
		};

		counts = hCount;
	}

	auto result = forGroups(counts);

	std::vector<unsigned int> expectedGroupNumbers = {
		0, 1, 1, 4, 4, 4, 4, 5
	};

	std::vector<unsigned int> expectedItemNumbers = {
		0, 0, 1, 0, 1, 2, 3, 0
	};

	thrust::host_vector<unsigned int> resultGroupNumbers = result->groupNumbers;
	thrust::host_vector<unsigned int> resultItemNumbers = result->itemNumbers;

	REQUIRE(std::equal(expectedGroupNumbers.begin(), expectedGroupNumbers.end(), resultGroupNumbers.begin()));
	REQUIRE(std::equal(expectedItemNumbers.begin(), expectedItemNumbers.end(), resultItemNumbers.begin()));
	REQUIRE(result->threadCount == expectedGroupNumbers.size());
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
		
		//a1 - b1
		//a2 - b2
		//a2 - c2
		//b1 - c1
		//b2 - c1
		//b2 - c3
		
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

		//a1 - b1
		//a2 - b2
		//a2 - c2
		//b1 - c1
		//b2 - c1
		//b2 - c3		

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


/*
Test for graph

		 C3-D2
		  |
A1-B1-C1-B2-A2-C2-D1
   
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | insert third level counts")
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

			c2 - d1
			c3 - d2

			a1-b1-c1  1
				  
			a2-b2 c1  2
				  c3

			b1-c1-	  0
			b2-c1-    0

			b2-c3-d2  1


		*/
		std::vector<FeatureInstance> hPairsA = {
			 { 0x000A0001 }
			,{ 0x000A0002 }
			,{ 0x000A0002 }

			,{ 0x000B0001 }
			,{ 0x000B0002 }
			,{ 0x000B0002 }

			,{ 0x000C0002 }
			,{ 0x000C0003 }
		};

		std::vector<FeatureInstance> hPairsB = {
			 { 0x000B0001 }
			,{ 0x000B0002 }
			,{ 0x000C0002 }

			,{ 0x000C0001 }
			,{ 0x000C0001 }
			,{ 0x000C0003 }

			,{ 0x000D0001 }
			,{ 0x000D0002 }
		};

		pairsA = hPairsA;
		pairsB = hPairsB;
	}

	auto instanceNeighboursMap = InstanceTypedNeighboursMapCreator::createTypedNeighboursListMap(
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

	thrust::device_vector<FeatureInstance> secondLevelInstances;
	{
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

		secondLevelInstances = expectedSecondLevel;
	}


	thrust::device_vector<unsigned int> group;
	thrust::device_vector<thrust::device_ptr<unsigned int>> groups;
	{
		//        g   i 
 		//a1 - b1   0   0
		//a2 - b2   0   1
		//a2 - c2   1   0
		//b1 - c1   2   0   
		//b2 - c1   2   1
		//b2 - c3   2   2
		

		std::vector<unsigned int> hgroups = { 0, 0, 1, 1, 1 };

		group = hgroups;
	}

	groups.push_back(group.data());
	groups.push_back(group.data());

	// ####################################################################

	const unsigned int outpuCount = secondLevelInstances.size();

	thrust::device_vector<unsigned int> thirdLevelCounts(outpuCount);

	thrust::device_vector<unsigned int> result(outpuCount);

	dim3 insertGrid;
	findSmallest2D(outpuCount, 256, insertGrid.x, insertGrid.y);

	thrust::device_vector<bool> integrityMask(outpuCount, true);

	fillWithNextLevelCountsFromTypedNeighbour <<< insertGrid, 256 >>> (
		instanceNeighboursMap->map->getBean()
		, thrust::raw_pointer_cast(cliques.data())
		, thrust::raw_pointer_cast(groups.data())
		, secondLevelInstances.data()
		, outpuCount
		, 2
		, integrityMask.data()
		, result.data()
	);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	std::vector<unsigned int> expectedCounts{ 1, 2, 0, 0, 1 };

	thrust::host_vector<unsigned int> resultCounts = result;

	REQUIRE(std::equal(expectedCounts.begin(), expectedCounts.end(), resultCounts.begin()));
}



/*
Test for graph

		 C3-D2
		 |
A1-B1-C1-B2-A2-C2-D1


a1-b1-c1  1

a2-b2 c1  2
	  c3

b1-c1-	  0
b2-c1-    0

b2-c3-d2  1
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | insert third..n level instances")
{
	// planessweep data
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
		c2 - d1
		c3 - d2
		*/
		std::vector<FeatureInstance> hPairsA = {
			{ 0x000A0001 }
			,{ 0x000A0002 }
			,{ 0x000A0002 }
			,{ 0x000B0001 }
			,{ 0x000B0002 }
			,{ 0x000B0002 }
			,{ 0x000C0002 }
			,{ 0x000C0003 }
		};

		std::vector<FeatureInstance> hPairsB = {
			{ 0x000B0001 }
			,{ 0x000B0002 }
			,{ 0x000C0002 }
			,{ 0x000C0001 }
			,{ 0x000C0001 }
			,{ 0x000C0003 }
			,{ 0x000D0001 }
			,{ 0x000D0002 }
		};

		pairsA = hPairsA;
		pairsB = hPairsB;
	}


	// instance neighbour map
	auto instanceNeighboursMap = InstanceTypedNeighboursMapCreator::createTypedNeighboursListMap(
		pairsA
		, pairsB
	);

	// clique data
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


	// forgroups result

	std::vector<UIntThrustVectorPtr> forGroupGroups;
	thrust::device_vector<thrust::device_ptr<unsigned int>> forGroupGroupsDevPtrs;
	{
		std::vector<thrust::device_ptr<unsigned int>> tempDevPtr;

		std::vector<unsigned int> hFirstLevelGroups = { 0, 0, 1, 1, 1 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hFirstLevelGroups));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		std::vector<unsigned int> hsecondLevelGroup = { 0, 0, 1, 1, 1 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hsecondLevelGroup));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		/*
			groups count 
			1, 2, 0, 0, 1
		*/
		std::vector<unsigned int> hthirdLevel = { 0, 1, 1, 4 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hthirdLevel));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		forGroupGroupsDevPtrs = tempDevPtr;
	}

	thrust::device_vector<unsigned int> itemsNumber;
	{
		std::vector<unsigned int> hItemumbers = { 0, 0, 1, 0 };
		itemsNumber = hItemumbers;
	}

	// last ready level instances
	thrust::device_vector<FeatureInstance> secondLevelInstances;
	{
		std::vector<FeatureInstance> hSecondLevelInstances;
		{
			FeatureInstance fi;

			fi.field = 0x000B0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0003;
			hSecondLevelInstances.push_back(fi);
		}

		secondLevelInstances = hSecondLevelInstances;
	}


	// ####################################################################
	
	const unsigned int outpuCount = forGroupGroups.back()->size();

	thrust::device_vector<FeatureInstance> result(outpuCount);

	dim3 insertGrid;
	findSmallest2D(outpuCount, 256, insertGrid.x, insertGrid.y);
	
	fillLevelInstancesFromNeighboursList << < insertGrid, 256 >> > (
		instanceNeighboursMap->map->getBean()
		, cliques.data().get()
		, forGroupGroupsDevPtrs.data().get()
		, itemsNumber.data()
		, secondLevelInstances.data()
		, pairsB.data()
		, outpuCount
		, 2
		, result.data()
		);
		
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	std::vector<FeatureInstance> expectedThirdLevelInstances;
	{ 
		FeatureInstance fi;
		/*
		a1-b1-c1  1

		a2-b2 c1  2
			  c3

		b1-c1-	  0
		b2-c1-    0

		b2-c3-d2  1
		*/

		fi.field = 0x000C0001;
		expectedThirdLevelInstances.push_back(fi);

		fi.field = 0x000C0001;
		expectedThirdLevelInstances.push_back(fi);

		fi.field = 0x000C0003;
		expectedThirdLevelInstances.push_back(fi);

		fi.field = 0x000D0002;
		expectedThirdLevelInstances.push_back(fi);
	}

	thrust::host_vector<FeatureInstance> calculatedThirdLevelInstances = result;

	REQUIRE(std::equal(expectedThirdLevelInstances.begin(), expectedThirdLevelInstances.end(), calculatedThirdLevelInstances.begin()));
}
// -------------------------------------------------------------------------------------------------------------------------------

/*
Test for graph

		 C3-D2
		 | /
A1-B1-C1-B2-A2-C2-D1

*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | check clique integrity")
{
	// planessweep data
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
		b2 - d2
		c2 - d1
		c3 - d2
		*/
		std::vector<FeatureInstance> hPairsA = {
			{ 0x000A0001 }
			,{ 0x000A0002 }
			,{ 0x000A0002 }
			,{ 0x000B0001 }
			,{ 0x000B0002 }
			,{ 0x000B0002 }
			,{ 0x000B0002 }
			,{ 0x000C0002 }
			,{ 0x000C0003 }
		};

		std::vector<FeatureInstance> hPairsB = {
			{ 0x000B0001 }
			,{ 0x000B0002 }
			,{ 0x000C0002 }
			,{ 0x000C0001 }
			,{ 0x000C0001 }
			,{ 0x000C0003 }
			,{ 0x000D0002 }
			,{ 0x000D0001 }
			,{ 0x000D0002 }
		};

		pairsA = hPairsA;
		pairsB = hPairsB;
	}


	// instance neighbour map
	auto instanceNeighboursMap = InstanceTypedNeighboursMapCreator::createTypedNeighboursListMap(
		pairsA
		, pairsB
	);

	// clique data
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


	// forgroups result

	std::vector<UIntThrustVectorPtr> forGroupGroups;
	thrust::device_vector<thrust::device_ptr<unsigned int>> forGroupGroupsDevPtrs;
	{
		std::vector<thrust::device_ptr<unsigned int>> tempDevPtr;

		std::vector<unsigned int> hFirstLevelGroups = { 0, 0, 1, 1, 1 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hFirstLevelGroups));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		std::vector<unsigned int> hsecondLevelGroup = { 0, 0, 1, 1, 1 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hsecondLevelGroup));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		/*
		groups count
		1, 2, 0, 0, 1
		*/
		std::vector<unsigned int> hthirdLevel = { 0, 1, 1, 4 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hthirdLevel));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		forGroupGroupsDevPtrs = tempDevPtr;
	}

	thrust::device_vector<unsigned int> itemsNumber;
	{
		std::vector<unsigned int> hItemumbers = { 0, 0, 1, 0 };
		itemsNumber = hItemumbers;
	}



	// instances levels
	thrust::device_vector<FeatureInstance> firstLevelInstances;
	{
		/*
		a1-b1-c1  1

		a2-b2 c1  2
			  c3

		b1-c1-	  0
		b2-c1-    0

		b2-c3-d2  1
		*/

		std::vector<FeatureInstance> hFirstLevelInstances;
		{
			FeatureInstance fi;

			fi.field = 0x000A0001;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000A0002;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000B0001;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hFirstLevelInstances.push_back(fi);
		}

		firstLevelInstances = hFirstLevelInstances;
	}

	thrust::device_vector<FeatureInstance> secondLevelInstances;
	{
		std::vector<FeatureInstance> hSecondLevelInstances;
		{
			FeatureInstance fi;

			fi.field = 0x000B0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0003;
			hSecondLevelInstances.push_back(fi);
		}

		secondLevelInstances = hSecondLevelInstances;
	}

	thrust::device_vector<FeatureInstance> thirdLevelInstances;
	{
		std::vector<FeatureInstance> hThirdLevelInstances;
		{
			FeatureInstance fi;

			fi.field = 0x000C0001;
			hThirdLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hThirdLevelInstances.push_back(fi);

			fi.field = 0x000C0003;
			hThirdLevelInstances.push_back(fi);

			fi.field = 0x000D0002;
			hThirdLevelInstances.push_back(fi);
		}

		thirdLevelInstances = hThirdLevelInstances;
	}

	thrust::device_vector<thrust::device_ptr<FeatureInstance>> instancesOnLevels;
	{
		std::vector<thrust::device_ptr<FeatureInstance>> hInstancesOnLevels;

		hInstancesOnLevels.push_back(firstLevelInstances.data());
		hInstancesOnLevels.push_back(secondLevelInstances.data());
		hInstancesOnLevels.push_back(thirdLevelInstances.data());

		instancesOnLevels = hInstancesOnLevels;
	}

	const unsigned int outpuCount = forGroupGroups.back()->size();

	thrust::device_vector<bool> result(outpuCount);

	dim3 insertGrid;
	findSmallest2D(outpuCount, 256, insertGrid.x, insertGrid.y);

	markAsPartOfCurrentCliqueInstance <<< insertGrid, 256 >>> (
		instanceNeighboursMap->map->getBean()
		, forGroupGroupsDevPtrs.data().get()
		, instancesOnLevels.data().get()
		, thirdLevelInstances.data()
		, pairsB.data()
		, outpuCount
		, 2
		, result.data()
	);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	std::vector<bool> expected = { false, false, false, true };
	thrust::host_vector<bool> calculated = result;

	REQUIRE(std::equal(expected.begin(), expected.end(), calculated.begin()));
}
// ---------------------------------------------------------------------------------------------


TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | reverese generate simple 1")
{
	// instances levels
	thrust::device_vector<FeatureInstance> firstLevelInstances;
	{
		std::vector<FeatureInstance> hFirstLevelInstances;
		{
			FeatureInstance fi;

			fi.field = 0x000A0001; 
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000A0002;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000B0001;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hFirstLevelInstances.push_back(fi);
		}

		firstLevelInstances = hFirstLevelInstances;
	}

	thrust::device_vector<FeatureInstance> secondLevelInstances;
	{
		std::vector<FeatureInstance> hSecondLevelInstances;
		{
			FeatureInstance fi;

			fi.field = 0x000B0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0003;
			hSecondLevelInstances.push_back(fi);
		}

		secondLevelInstances = hSecondLevelInstances;
	}

	thrust::device_vector<FeatureInstance> thirdLevelInstances;
	{
		std::vector<FeatureInstance> hThirdLevelInstances;
		{
			FeatureInstance fi;

			fi.field = 0x000C0001;
			hThirdLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hThirdLevelInstances.push_back(fi);

			fi.field = 0x000C0003;
			hThirdLevelInstances.push_back(fi);

			fi.field = 0x000D0002;
			hThirdLevelInstances.push_back(fi);
		}

		thirdLevelInstances = hThirdLevelInstances;
	}

	thrust::device_vector<thrust::device_ptr<FeatureInstance>> instancesOnLevels;
	{
		std::vector<thrust::device_ptr<FeatureInstance>> hInstancesOnLevels;

		hInstancesOnLevels.push_back(firstLevelInstances.data());
		hInstancesOnLevels.push_back(secondLevelInstances.data());
		hInstancesOnLevels.push_back(thirdLevelInstances.data());

		instancesOnLevels = hInstancesOnLevels;
	}


	// forgroups result

	std::vector<UIntThrustVectorPtr> forGroupGroups;
	thrust::device_vector<thrust::device_ptr<unsigned int>> forGroupGroupsDevPtrs;
	{
		std::vector<thrust::device_ptr<unsigned int>> tempDevPtr;

		std::vector<unsigned int> hFirstLevelGroups = { 0, 0, 1, 1, 1 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hFirstLevelGroups));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		std::vector<unsigned int> hsecondLevelGroup = { 0, 0, 1, 1, 1 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hsecondLevelGroup));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		/*
		groups count
		1, 2, 0, 0, 1
		*/
		std::vector<unsigned int> hthirdLevel = { 0, 1, 1, 4 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hthirdLevel));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		forGroupGroupsDevPtrs = tempDevPtr;
	}
	
	// write positions
	thrust::device_vector<unsigned int> writePositions;
	{
		std::vector<unsigned int> hWritePositions = { 0, 1, 1, 2 };
		writePositions = hWritePositions;			
	}

	// integrity mask
	thrust::device_vector<bool> integrityMask;
	{
		std::vector<bool> hIntegrityMask = { true, true, false, true };
		integrityMask = hIntegrityMask;
	}

	// result
	
	const unsigned int endCount = 4;
	const unsigned int consistentCount = 3;
	const unsigned int cliqueSize = 3;

	thrust::device_vector<FeatureInstance> result(consistentCount * cliqueSize);

	dim3 insertGrid;
	findSmallest2D(endCount, 256, insertGrid.x, insertGrid.y);

	reverseGenerateCliquesInstances <<< insertGrid, 256 >>> (
		forGroupGroupsDevPtrs.data().get()
		, instancesOnLevels.data().get()
		, endCount
		, consistentCount
		, cliqueSize
		, integrityMask.data()
		, writePositions.data()
		, result.data()
	);


	/*
	a1-b1-c1  1

	a2-b2 c1  2
		  c3

	b1-c1-	  0
	b2-c1-    0

	b2-c3-d2  1
	*/

	std::vector<FeatureInstance> expected = {
		 { 0x000A0001 }, { 0x000A0002 }, { 0x000B0002 }
		,{ 0x000B0001 }, { 0x000B0002 }, { 0x000C0003 }
		,{ 0x000C0001 }, { 0x000C0001 }, { 0x000D0002 }
	};

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	thrust::host_vector<FeatureInstance> copmuted = result;

	REQUIRE(std::equal(expected.begin(), expected.end(), copmuted.begin()));
}


TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | reverese generate simple 2")
{
	// instances levels
	thrust::device_vector<FeatureInstance> firstLevelInstances;
	{
		std::vector<FeatureInstance> hFirstLevelInstances;
		{
			FeatureInstance fi;

			fi.field = 0x000A0001;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000A0002;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000B0001;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hFirstLevelInstances.push_back(fi);
		}

		firstLevelInstances = hFirstLevelInstances;
	}

	thrust::device_vector<FeatureInstance> secondLevelInstances;
	{
		std::vector<FeatureInstance> hSecondLevelInstances;
		{
			FeatureInstance fi;

			fi.field = 0x000B0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0003;
			hSecondLevelInstances.push_back(fi);
		}

		secondLevelInstances = hSecondLevelInstances;
	}

	thrust::device_vector<FeatureInstance> thirdLevelInstances;
	{
		std::vector<FeatureInstance> hThirdLevelInstances;
		{
			FeatureInstance fi;

			fi.field = 0x000C0001;
			hThirdLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hThirdLevelInstances.push_back(fi);

			fi.field = 0x000C0003;
			hThirdLevelInstances.push_back(fi);

			fi.field = 0x000D0002;
			hThirdLevelInstances.push_back(fi);
		}

		thirdLevelInstances = hThirdLevelInstances;
	}

	thrust::device_vector<thrust::device_ptr<FeatureInstance>> instancesOnLevels;
	{
		std::vector<thrust::device_ptr<FeatureInstance>> hInstancesOnLevels;

		hInstancesOnLevels.push_back(firstLevelInstances.data());
		hInstancesOnLevels.push_back(secondLevelInstances.data());
		hInstancesOnLevels.push_back(thirdLevelInstances.data());

		instancesOnLevels = hInstancesOnLevels;
	}


	// forgroups result

	std::vector<UIntThrustVectorPtr> forGroupGroups;
	thrust::device_vector<thrust::device_ptr<unsigned int>> forGroupGroupsDevPtrs;
	{
		std::vector<thrust::device_ptr<unsigned int>> tempDevPtr;

		std::vector<unsigned int> hFirstLevelGroups = { 0, 0, 1, 1, 1 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hFirstLevelGroups));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		std::vector<unsigned int> hsecondLevelGroup = { 0, 0, 1, 1, 1 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hsecondLevelGroup));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		/*
		groups count
		1, 2, 0, 0, 1
		*/
		std::vector<unsigned int> hthirdLevel = { 0, 1, 1, 4 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hthirdLevel));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		forGroupGroupsDevPtrs = tempDevPtr;
	}

	// write positions
	thrust::device_vector<unsigned int> writePositions;
	{
		std::vector<unsigned int> hWritePositions = { 0, 1, 1, 2 };
		writePositions = hWritePositions;
	}

	// integrity mask
	thrust::device_vector<bool> integrityMask;
	{
		std::vector<bool> hIntegrityMask = { true, false, true, true };
		integrityMask = hIntegrityMask;
	}

	// result

	const unsigned int endCount = 4;
	const unsigned int consistentCount = 3;
	const unsigned int cliqueSize = 3;

	thrust::device_vector<FeatureInstance> result(consistentCount * cliqueSize);

	dim3 insertGrid;
	findSmallest2D(endCount, 256, insertGrid.x, insertGrid.y);

	reverseGenerateCliquesInstances << < insertGrid, 256 >> > (
		forGroupGroupsDevPtrs.data().get()
		, instancesOnLevels.data().get()
		, endCount
		, consistentCount
		, cliqueSize
		, integrityMask.data()
		, writePositions.data()
		, result.data()
		);


	/*
	a1-b1-c1  1

	a2-b2 c1  2
	      c3

	b1-c1-	  0
	b2-c1-    0

	b2-c3-d2  1
	*/

	std::vector<FeatureInstance> expected = {
		{ 0x000A0001 },{ 0x000A0002 },{ 0x000B0002 }
		,{ 0x000B0001 },{ 0x000B0002 },{ 0x000C0003 }
		,{ 0x000C0001 },{ 0x000C0003 },{ 0x000D0002 }
	};

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	thrust::host_vector<FeatureInstance> copmuted = result;

	REQUIRE(std::equal(expected.begin(), expected.end(), copmuted.begin()));
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | reverese generate simple 2-size")
{
	// instances levels
	thrust::device_vector<FeatureInstance> firstLevelInstances;
	{
		std::vector<FeatureInstance> hFirstLevelInstances;
		{
			FeatureInstance fi;

			fi.field = 0x000A0001;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000A0002;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000B0001;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hFirstLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hFirstLevelInstances.push_back(fi);
		}

		firstLevelInstances = hFirstLevelInstances;
	}

	thrust::device_vector<FeatureInstance> secondLevelInstances;
	{
		std::vector<FeatureInstance> hSecondLevelInstances;
		{
			FeatureInstance fi;

			fi.field = 0x000B0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000B0002;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0001;
			hSecondLevelInstances.push_back(fi);

			fi.field = 0x000C0003;
			hSecondLevelInstances.push_back(fi);
		}

		secondLevelInstances = hSecondLevelInstances;
	}


	thrust::device_vector<thrust::device_ptr<FeatureInstance>> instancesOnLevels;
	{
		std::vector<thrust::device_ptr<FeatureInstance>> hInstancesOnLevels;

		hInstancesOnLevels.push_back(firstLevelInstances.data());
		hInstancesOnLevels.push_back(secondLevelInstances.data());

		instancesOnLevels = hInstancesOnLevels;
	}


	// forgroups result

	std::vector<UIntThrustVectorPtr> forGroupGroups;
	thrust::device_vector<thrust::device_ptr<unsigned int>> forGroupGroupsDevPtrs;
	{
		std::vector<thrust::device_ptr<unsigned int>> tempDevPtr;

		std::vector<unsigned int> hFirstLevelGroups = { 0, 0, 1, 1, 1 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hFirstLevelGroups));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		std::vector<unsigned int> hsecondLevelGroup = { 0, 0, 1, 1, 1 };

		forGroupGroups.push_back(std::make_shared<UIntThrustVector>(hsecondLevelGroup));
		tempDevPtr.push_back(forGroupGroups.back()->data());

		forGroupGroupsDevPtrs = tempDevPtr;
	}

	// write positions
	thrust::device_vector<unsigned int> writePositions;
	{
		std::vector<unsigned int> hWritePositions = { 0, 1, 1, 2, 3 };
		writePositions = hWritePositions;
	}

	// integrity mask
	thrust::device_vector<bool> integrityMask;
	{
		std::vector<bool> hIntegrityMask = { true, false, true, true, true };
		integrityMask = hIntegrityMask;
	}

	// result

	const unsigned int endCount = 5;
	const unsigned int consistentCount = 4;
	const unsigned int cliqueSize = 2;

	thrust::device_vector<FeatureInstance> result(consistentCount * cliqueSize);

	dim3 insertGrid;
	findSmallest2D(endCount, 256, insertGrid.x, insertGrid.y);

	reverseGenerateCliquesInstances << < insertGrid, 256 >> > (
		forGroupGroupsDevPtrs.data().get()
		, instancesOnLevels.data().get()
		, endCount
		, consistentCount
		, cliqueSize
		, integrityMask.data()
		, writePositions.data()
		, result.data()
		);


	/*
	a1-b1  true
					a2-b2  false
	b1-c1  true
	b2-c1  true
	b2-c3  true
	*/

	std::vector<FeatureInstance> expected = {
		 { 0x000A0001 },{ 0x000B0001 },{ 0x000B0002 },{ 0x000B0002 }
		,{ 0x000B0001 },{ 0x000C0001 },{ 0x000C0001 },{ 0x000C0003 }
	};

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	thrust::host_vector<FeatureInstance> copmuted = result;

	REQUIRE(std::equal(expected.begin(), expected.end(), copmuted.begin()));
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | generate write positions")
{
	thrust::device_vector<bool> integrityMask;
	{
		std::vector<bool> hIntegrityMask{ true, true, true, false, false, true };
		integrityMask = hIntegrityMask;
	}

	thrust::device_vector<unsigned int> writePositions(6);

	auto consistentCount = fillWritePositionsAndReturnCount(
		integrityMask
		, writePositions
		, 6
	);

	std::vector<unsigned int> expected = { 0, 1, 2, 3, 3, 3 };

	thrust::host_vector<unsigned int> calculated = writePositions;

	REQUIRE(consistentCount == 4);
	REQUIRE(std::equal(expected.begin(), expected.end(), calculated.begin()));
}


TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree helpers | generate write positions, redundant integrity mask")
{
	thrust::device_vector<bool> integrityMask;
	{
		std::vector<bool> hIntegrityMask{ true, true, true, false, false, true, false, false, true };
		integrityMask = hIntegrityMask;
	}

	thrust::device_vector<unsigned int> writePositions(6);

	auto consistentCount = fillWritePositionsAndReturnCount(
		integrityMask
		, writePositions
		, 6
	);

	std::vector<unsigned int> expected = { 0, 1, 2, 3, 3, 3 };

	thrust::host_vector<unsigned int> calculated = writePositions;

	REQUIRE(consistentCount == 4);
	REQUIRE(std::equal(expected.begin(), expected.end(), calculated.begin()));
}