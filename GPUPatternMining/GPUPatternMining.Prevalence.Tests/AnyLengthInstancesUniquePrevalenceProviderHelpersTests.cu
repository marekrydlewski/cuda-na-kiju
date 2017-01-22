#include "catch.hpp"

#include <map> 
#include <thrust/device_vector.h>
#include <thrust/unique.h>
//----------------------------------------------------------------------------------------


#define TEST_CUDA_CHECK_RETURN
//----------------------------------------------------------------------------------------

#include "BaseCudaTestHandler.h"
#include "../GPUPatternMining/Entities/TypeCount.h"
#include "../GPUPatternMining/Prevalence/AnyLengthInstancesUniquePrevalenceProvider.h"
//----------------------------------------------------------------------------------------


TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProviderHelpers | AnyLengthInstancesUniquePrevalenceProvider 1")
{
	auto counts = std::make_shared<TypesCounts>();
	counts->push_back(TypeCount(0xA, 2));
	counts->push_back(TypeCount(0xB, 4));
	counts->push_back(TypeCount(0xC, 2));

	GPUKeyProcessor<unsigned int> key;

	auto result = getGpuTypesCountsMap(counts, &key);


	std::vector<unsigned int> lKeys = { 0xA, 0xB, 0xC };
	thrust::device_vector<unsigned int> keys = lKeys;

	thrust::device_vector<unsigned int> values(3);

	result->map->getValues(
		keys.data().get()
		, values.data().get()
		, 3
	);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	std::vector<unsigned int> expected = { 2, 4, 2 };
	thrust::host_vector<unsigned int> calculated = values;

	REQUIRE(std::equal(expected.begin(), expected.end(), calculated.begin()));
}


TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProviderHelpers | moveCliquesCandidatesToGpu 1")
{
	CliquesCandidates candidates = {
		{ 0xA, 0xB }
		,{ 0xA, 0xC }
		,{ 0xB, 0xC }
	};

	auto gpuCandidates = Entities::moveCliquesCandidatesToGpu(candidates);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	REQUIRE(gpuCandidates.thrustVectorsOfCliques->size() == 3);
	REQUIRE(gpuCandidates.candidatesCount == 3);
	REQUIRE(gpuCandidates.currentCliquesSize == 2);

	std::vector<std::vector<unsigned short>> expected = 
	{
		{ 0xA, 0xB }
		,{ 0xA, 0xC }
		,{ 0xB, 0xC }
	};

	for (unsigned int i = 0; i < 3; ++i)
	{
		thrust::host_vector<unsigned short> currentRes = (*gpuCandidates.thrustVectorsOfCliques->at(i));

		REQUIRE(std::equal(
			expected[i].begin()
			, expected[i].end()
			, currentRes.begin()));
	}
}

TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProviderHelpers | getTypesCountOnGpuForCliquesCandidates 1")
{
	// create candidates
	CliquesCandidates candidates = {
		{ 0xA, 0xB }
		,{ 0xA, 0xC }
		,{ 0xB, 0xC }
	};

	auto gpuCandidates = Entities::moveCliquesCandidatesToGpu(candidates);

	const unsigned int candidatesCount = gpuCandidates.candidatesCount;

	// create count
	auto counts = std::make_shared<TypesCounts>();
	counts->push_back(TypeCount(0xA, 2));
	counts->push_back(TypeCount(0xB, 4));
	counts->push_back(TypeCount(0xC, 2));

	GPUKeyProcessor<unsigned int> key;

	auto typesCountsMap = getGpuTypesCountsMap(counts, &key);

	// perform

	auto cliquesTypesCount = getTypesCountOnGpuForCliquesCandidates(
		gpuCandidates, typesCountsMap->map
	);

	// check

	std::vector<unsigned int> expected = { 2, 2, 4, 4, 2, 2 };
	thrust::host_vector<unsigned int> calculated = (*cliquesTypesCount);

	REQUIRE(std::equal(expected.begin(), expected.end(), calculated.begin()));

}

TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProviderHelpers | MinimalCandidatePrevalenceCounter 1")
{

}