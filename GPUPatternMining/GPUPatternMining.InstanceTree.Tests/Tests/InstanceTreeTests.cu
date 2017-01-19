#include "..\catch.hpp"
#include "..\BaseCudaTestHandler.h"

#define TEST_CUDA_CHECK_RETURN
//--------------------------------------------------------------

#include "..\..\GPUPatternMining\InstanceTree\InstanceTree.h"
//--------------------------------------------------------------

/*
	B0-A0-C0

	B1-C1

*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree | single size-2")
{
	PlaneSweepTableInstanceResultPtr planeSweepResult = std::make_shared<PlaneSweepTableInstanceResult>();
	
	// pairs
	{
		std::vector<FeatureInstance> pairA = {
			 { 0x000A0000 }
			,{ 0x000A0000 }
			,{ 0x000B0001 }
		};

		planeSweepResult->pairsA = pairA;

		std::vector<FeatureInstance> pairB = {
			 { 0x000B0000 }
			,{ 0x000C0000 }
			,{ 0x000C0001 }
		};

		planeSweepResult->pairsB = pairB;
	}

	IntanceTablesMapCreator::ITMPackPtr typedPairInstancesPack = IntanceTablesMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	InstanceTypedNeighboursMapCreator::ITNMPackPtr instanceTypedNeighbursPack = InstanceTypedNeighboursMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	auto instanceTree = InstanceTree::InstanceTree(
		planeSweepResult
		, typedPairInstancesPack
		, instanceTypedNeighbursPack);

	Entities::CliquesCandidates cc = {
		{ 0x000A, 0x000B }
	};

	auto ccOnGpu = Entities::moveCliquesCandidatesToGpu(cc);
	auto result = instanceTree.getInstancesResult(ccOnGpu);

	thrust::host_vector<FeatureInstance> calculated = result->instances;

	std::vector<FeatureInstance> expected = { { 0x000A0000 }, { 0x000B0000 } };
	REQUIRE(std::equal(expected.begin(), expected.end(), calculated.begin()));

	std::vector<unsigned int> expectedInstancesId = { 0 };
	thrust::host_vector<unsigned int> calculatedCliqueId = result->instancesCliqueId;
	REQUIRE(std::equal(expectedInstancesId.begin(), expectedInstancesId.end(), calculatedCliqueId.begin()));
}


/*
B0-A0-C0

B1-C1

*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree | multiple size-2")
{
	PlaneSweepTableInstanceResultPtr planeSweepResult = std::make_shared<PlaneSweepTableInstanceResult>();

	// pairs
	{
		std::vector<FeatureInstance> pairA = {
			{ 0x000A0000 }
			,{ 0x000A0000 }
			,{ 0x000B0001 }
		};

		planeSweepResult->pairsA = pairA;

		std::vector<FeatureInstance> pairB = {
			{ 0x000B0000 }
			,{ 0x000C0000 }
			,{ 0x000C0001 }
		};

		planeSweepResult->pairsB = pairB;
	}

	IntanceTablesMapCreator::ITMPackPtr typedPairInstancesPack = IntanceTablesMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	InstanceTypedNeighboursMapCreator::ITNMPackPtr instanceTypedNeighbursPack = InstanceTypedNeighboursMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	auto instanceTree = InstanceTree::InstanceTree(
		planeSweepResult
		, typedPairInstancesPack
		, instanceTypedNeighbursPack);

	Entities::CliquesCandidates cc = {
		{ 0x000A, 0x000B }
		, { 0x000A, 0x000C }
		, { 0x000B, 0x000C }
	};

	auto ccOnGpu = Entities::moveCliquesCandidatesToGpu(cc);
	auto result = instanceTree.getInstancesResult(ccOnGpu);

	thrust::host_vector<FeatureInstance> calculated = result->instances;

	std::vector<FeatureInstance> expected = { 
		 { 0x000A0000 },{ 0x000A0000 },{ 0x000B0001 }
		,{ 0x000B0000 },{ 0x000C0000 },{ 0x000C0001 }
	};

	REQUIRE(std::equal(expected.begin(), expected.end(), calculated.begin()));

	std::vector<unsigned int> expectedInstancesId = { 0, 1, 2 };
	thrust::host_vector<unsigned int> calculatedCliqueId = result->instancesCliqueId;
	REQUIRE(std::equal(expectedInstancesId.begin(), expectedInstancesId.end(), calculatedCliqueId.begin()));

}

/*

	   A0-C0
		\ /
		B0
		/ \
	   D0-C1
		\ /
		 B1
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree | single size-3")
{
	PlaneSweepTableInstanceResultPtr planeSweepResult = std::make_shared<PlaneSweepTableInstanceResult>();

	// pairs
	{
		std::vector<FeatureInstance> pairA = {
			 { 0xA0000 }
			,{ 0xA0000 }
			,{ 0xB0000 }
			,{ 0xB0000 }
			,{ 0xB0000 }
			,{ 0xB0001 }
			,{ 0xB0001 }
			,{ 0xC0001 }
		};

		planeSweepResult->pairsA = pairA;

		std::vector<FeatureInstance> pairB = {
			 { 0xB0000 }
			,{ 0xC0000 }
			,{ 0xC0000 }
			,{ 0xC0001 }
			,{ 0xD0000 }
			,{ 0xC0001 }
			,{ 0xD0000 }
			,{ 0xD0000 }
		};

		planeSweepResult->pairsB = pairB;
	}

	IntanceTablesMapCreator::ITMPackPtr typedPairInstancesPack = IntanceTablesMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	InstanceTypedNeighboursMapCreator::ITNMPackPtr instanceTypedNeighbursPack = InstanceTypedNeighboursMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	auto instanceTree = InstanceTree::InstanceTree(
		planeSweepResult
		, typedPairInstancesPack
		, instanceTypedNeighbursPack);

	Entities::CliquesCandidates cc = {
		{ 0x000A, 0x000B, 0x000C }
	};

	auto ccOnGpu = Entities::moveCliquesCandidatesToGpu(cc);
	auto result = instanceTree.getInstancesResult(ccOnGpu);

	thrust::host_vector<FeatureInstance> calculated = result->instances;

	std::vector<FeatureInstance> expected = {
		{ 0xA0000 },{ 0xB0000 },{ 0xC0000 }
	};

	REQUIRE(std::equal(expected.begin(), expected.end(), calculated.begin()));

	std::vector<unsigned int> expectedInstancesId = { 0 };
	thrust::host_vector<unsigned int> calculatedCliqueId = result->instancesCliqueId;
	REQUIRE(std::equal(expectedInstancesId.begin(), expectedInstancesId.end(), calculatedCliqueId.begin()));

}

/*

   A0-C0
	\ /
	B0
	/ \
   D0-C1
	\ /
	 B1
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree | multiple size-3")
{
	PlaneSweepTableInstanceResultPtr planeSweepResult = std::make_shared<PlaneSweepTableInstanceResult>();

	// pairs
	{
		std::vector<FeatureInstance> pairA = {
			{ 0xA0000 }
			,{ 0xA0000 }
			,{ 0xB0000 }
			,{ 0xB0000 }
			,{ 0xB0001 }
			,{ 0xB0000 }
			,{ 0xB0001 }
			,{ 0xC0001 }
		};

		planeSweepResult->pairsA = pairA;

		std::vector<FeatureInstance> pairB = {
			{ 0xB0000 }
			,{ 0xC0000 }
			,{ 0xC0000 }
			,{ 0xC0001 }
			,{ 0xC0001 }
			,{ 0xD0000 }
			,{ 0xD0000 }
			,{ 0xD0000 }
		};

		planeSweepResult->pairsB = pairB;
	}

	IntanceTablesMapCreator::ITMPackPtr typedPairInstancesPack = IntanceTablesMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	InstanceTypedNeighboursMapCreator::ITNMPackPtr instanceTypedNeighbursPack = InstanceTypedNeighboursMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	auto instanceTree = InstanceTree::InstanceTree(
		planeSweepResult
		, typedPairInstancesPack
		, instanceTypedNeighbursPack);

	Entities::CliquesCandidates cc = {
		 { 0x000A, 0x000B, 0x000C }
		,{ 0x000B, 0x000C, 0x000D }
	};

	auto ccOnGpu = Entities::moveCliquesCandidatesToGpu(cc);
	auto result = instanceTree.getInstancesResult(ccOnGpu);

	thrust::host_vector<FeatureInstance> calculated = result->instances;

	std::vector<FeatureInstance> expected = {
		 { 0xA0000 },{ 0xB0000 },{ 0xB0001 }
		,{ 0xB0000 },{ 0xC0001 },{ 0xC0001 }
		,{ 0xC0000 },{ 0xD0000 },{ 0xD0000 }
	};

	REQUIRE(std::equal(expected.begin(), expected.end(), calculated.begin()));

	std::vector<unsigned int> expectedInstancesId = { 0, 1, 1 };
	thrust::host_vector<unsigned int> calculatedCliqueId = result->instancesCliqueId;
	REQUIRE(std::equal(expectedInstancesId.begin(), expectedInstancesId.end(), calculatedCliqueId.begin()));

}

/*
Test for graph

clique (A0,B0,C0,D0,E0)

*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree | single size-5")
{
	PlaneSweepTableInstanceResultPtr planeSweepResult = std::make_shared<PlaneSweepTableInstanceResult>();

	// pairs
	{
		std::vector<FeatureInstance> pairA = {
			{ 0xA0000 }
			,{ 0xA0000 }
			,{ 0xA0000 }
			,{ 0xA0000 }

			,{ 0xB0000 }
			,{ 0xB0000 }
			,{ 0xB0000 }

			,{ 0xC0000 }
			,{ 0xC0000 }

			,{ 0xD0000 }
		};

		planeSweepResult->pairsA = pairA;

		std::vector<FeatureInstance> pairB = {
			{ 0xB0000 }
			,{ 0xC0000 }
			,{ 0xD0000 }
			,{ 0xE0000 }

			,{ 0xC0000 }
			,{ 0xD0000 }
			,{ 0xE0000 }

			,{ 0xD0000 }
			,{ 0xE0000 }

			,{ 0xE0000 }
		};

		planeSweepResult->pairsB = pairB;
	}

	IntanceTablesMapCreator::ITMPackPtr typedPairInstancesPack = IntanceTablesMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	InstanceTypedNeighboursMapCreator::ITNMPackPtr instanceTypedNeighbursPack = InstanceTypedNeighboursMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	auto instanceTree = InstanceTree::InstanceTree(
		planeSweepResult
		, typedPairInstancesPack
		, instanceTypedNeighbursPack);

	Entities::CliquesCandidates cc = {
		{ 0x000A, 0x000B, 0x000C, 0x000D, 0x000E }
	};

	auto ccOnGpu = Entities::moveCliquesCandidatesToGpu(cc);
	auto result = instanceTree.getInstancesResult(ccOnGpu);

	thrust::host_vector<FeatureInstance> calculated = result->instances;

	std::vector<FeatureInstance> expected = {
		{ 0xA0000 },{ 0xB0000 },{ 0xC0000 },{ 0xD0000 },{ 0xE0000 }
	};

	REQUIRE(std::equal(expected.begin(), expected.end(), calculated.begin()));

	std::vector<unsigned int> expectedInstancesId = { 0 };
	thrust::host_vector<unsigned int> calculatedCliqueId = result->instancesCliqueId;
	REQUIRE(std::equal(expectedInstancesId.begin(), expectedInstancesId.end(), calculatedCliqueId.begin()));

}

/*
 neighbour tree (graph looks very different), a0 - root
    
	b2-e1
   /        c2-e0
  /        /
a0-------b0-----c0---d0 T
  \        \      \--d1 F
   b1       c1     \-d2 F
                    \d3 T

*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree | complex 1")
{
	PlaneSweepTableInstanceResultPtr planeSweepResult = std::make_shared<PlaneSweepTableInstanceResult>();

	// pairs
	{
		std::vector<FeatureInstance> pairA = {

			{ 0xA0000 }
			,{ 0xA0000 }
			,{ 0xA0000 }

			,{ 0xA0000 }
			,{ 0xA0000 }
			,{ 0xA0000 }

			,{ 0xA0000 }
			,{ 0xA0000 }
			,{ 0xA0000 }

			,{ 0xB0000 }
			,{ 0xB0000 }
			,{ 0xB0000 }

			,{ 0xB0000 }
			,{ 0xB0000 }

			,{ 0xB0002 }

			,{ 0xC0000 }
			,{ 0xC0000 }
			,{ 0xC0000 }
			,{ 0xC0000 }

			,{ 0xC0002 }
		};

		planeSweepResult->pairsA = pairA;

		std::vector<FeatureInstance> pairB = {
			 { 0xB0000 }
			,{ 0xB0001 }
			,{ 0xB0002 }

			,{ 0xC0000 }
			,{ 0xC0001 }
			,{ 0xC0002 }

			,{ 0xD0000 }
			,{ 0xD0002 }
			,{ 0xD0003 }

			,{ 0xC0000 }
			,{ 0xC0001 }
			,{ 0xC0002 }

			,{ 0xD0000 }
			,{ 0xD0003 }

			,{ 0xE0001 }

			,{ 0xD0000 }
			,{ 0xD0001 }
			,{ 0xD0002 }
			,{ 0xD0003 }

			,{ 0xE0000 }
		};

		planeSweepResult->pairsB = pairB;
	}

	IntanceTablesMapCreator::ITMPackPtr typedPairInstancesPack = IntanceTablesMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	InstanceTypedNeighboursMapCreator::ITNMPackPtr instanceTypedNeighbursPack = InstanceTypedNeighboursMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	auto instanceTree = InstanceTree::InstanceTree(
		planeSweepResult
		, typedPairInstancesPack
		, instanceTypedNeighbursPack);

	Entities::CliquesCandidates cc = {
		{ 0x000A, 0x000B, 0x000C, 0x000D}
	};

	auto ccOnGpu = Entities::moveCliquesCandidatesToGpu(cc);
	auto result = instanceTree.getInstancesResult(ccOnGpu);

	thrust::host_vector<FeatureInstance> calculated = result->instances;

	std::vector<FeatureInstance> expected = {
		{ 0xA0000 },{ 0xA0000 }
		,{ 0xB0000 },{ 0xB0000 }
		,{ 0xC0000 },{ 0xC0000 }
		,{ 0xD0000 },{ 0xD0003 }
	};

	REQUIRE(std::equal(expected.begin(), expected.end(), calculated.begin()));

	std::vector<unsigned int> expectedInstancesId = { 0, 0 };
	thrust::host_vector<unsigned int> calculatedCliqueId = result->instancesCliqueId;
	REQUIRE(std::equal(expectedInstancesId.begin(), expectedInstancesId.end(), calculatedCliqueId.begin()));

}

/*
neighbour tree (graph looks very different), a0 - root

b2-e1
/        c2-e0
/        /
a0-------b0-----c0---d0 T
\        \      \--d1 F
b1       c1     \-d2 F
\d3 T

*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Instance tree | complex 2 (zero candidate instances)")
{
	PlaneSweepTableInstanceResultPtr planeSweepResult = std::make_shared<PlaneSweepTableInstanceResult>();

	// pairs
	{
		std::vector<FeatureInstance> pairA = {

			{ 0xA0000 }
			,{ 0xA0000 }
			,{ 0xA0000 }

			,{ 0xA0000 }
			,{ 0xA0000 }
			,{ 0xA0000 }

			,{ 0xA0000 }
			,{ 0xA0000 }
			,{ 0xA0000 }

			,{ 0xB0000 }
			,{ 0xB0000 }
			,{ 0xB0000 }

			,{ 0xB0000 }
			,{ 0xB0000 }

			,{ 0xB0002 }

			,{ 0xC0000 }
			,{ 0xC0000 }
			,{ 0xC0000 }
			,{ 0xC0000 }

			,{ 0xC0002 }
		};

		planeSweepResult->pairsA = pairA;

		std::vector<FeatureInstance> pairB = {
			{ 0xB0000 }
			,{ 0xB0001 }
			,{ 0xB0002 }

			,{ 0xC0000 }
			,{ 0xC0001 }
			,{ 0xC0002 }

			,{ 0xD0000 }
			,{ 0xD0002 }
			,{ 0xD0003 }

			,{ 0xC0000 }
			,{ 0xC0001 }
			,{ 0xC0002 }

			,{ 0xD0000 }
			,{ 0xD0003 }

			,{ 0xE0001 }

			,{ 0xD0000 }
			,{ 0xD0001 }
			,{ 0xD0002 }
			,{ 0xD0003 }

			,{ 0xE0000 }
		};

		planeSweepResult->pairsB = pairB;
	}

	IntanceTablesMapCreator::ITMPackPtr typedPairInstancesPack = IntanceTablesMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	InstanceTypedNeighboursMapCreator::ITNMPackPtr instanceTypedNeighbursPack = InstanceTypedNeighboursMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	auto instanceTree = InstanceTree::InstanceTree(
		planeSweepResult
		, typedPairInstancesPack
		, instanceTypedNeighbursPack);

	Entities::CliquesCandidates cc = {
		{ 0x000A, 0x000B, 0x000C, 0x000D, 0x000E }
	};

	auto ccOnGpu = Entities::moveCliquesCandidatesToGpu(cc);
	auto result = instanceTree.getInstancesResult(ccOnGpu);

	thrust::host_vector<FeatureInstance> calculated = result->instances;

	REQUIRE(calculated.size() == 0);
}