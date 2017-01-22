#include "catch.hpp"

#include <thrust/device_vector.h>
//----------------------------------------------------------------------------------------


#define TEST_CUDA_CHECK_RETURN
//----------------------------------------------------------------------------------------

#include "BaseCudaTestHandler.h"
#include "../GPUPatternMining/Entities/TypeCount.h"
#include "../GPUPatternMining/Prevalence/AnyLengthInstancesUniquePrevalenceProvider.h"
//----------------------------------------------------------------------------------------


TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProvider | init")
{
	auto counts = std::make_shared<TypesCounts>();
	counts->push_back(TypeCount(0xA, 2));
	counts->push_back(TypeCount(0xB, 2));
	counts->push_back(TypeCount(0xC, 2));

	auto keyProc = std::make_shared<GPUUIntKeyProcessor>();

	auto typesCountsMap = getGpuTypesCountsMap(counts, keyProc.get());

	auto counter = AnyLengthInstancesUniquePrevalenceProvider(typesCountsMap);
}

TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProvider | simple 0")
{
	auto counts = std::make_shared<TypesCounts>();
	counts->push_back(TypeCount(0xA, 2));
	counts->push_back(TypeCount(0xB, 2));
	counts->push_back(TypeCount(0xC, 2));

	auto keyProc = std::make_shared<GPUUIntKeyProcessor>();

	auto typesCountsMap = getGpuTypesCountsMap(counts, keyProc.get());

	auto counter = AnyLengthInstancesUniquePrevalenceProvider(typesCountsMap);

	CliquesCandidates candidates = {
		{0xA, 0xB}
	};

	auto gpuCandidates = Entities::moveCliquesCandidatesToGpu(candidates);

	auto instanceTreeResult = std::make_shared<InstanceTree::InstanceTreeResult>();

	std::vector<FeatureInstance> itInstances = { {0xA0000},{ 0xB0000 } };
	std::vector<unsigned int> itCliqueId = { 0 };

	instanceTreeResult->instances = itInstances;
	instanceTreeResult->instancesCliqueId = itCliqueId;

	auto result = counter.getPrevalenceFromCandidatesInstances(
		gpuCandidates
		, instanceTreeResult
	);

	std::vector<float> expected = { 0.5f };

	thrust::host_vector<float> hResult = *result;
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	REQUIRE(std::equal(expected.begin(), expected.end(), hResult.begin()));
}


TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProvider | simple 1")
{
	auto counts = std::make_shared<TypesCounts>();
	counts->push_back(TypeCount(0xA, 2));
	counts->push_back(TypeCount(0xB, 2));
	counts->push_back(TypeCount(0xC, 2));

	auto keyProc = std::make_shared<GPUUIntKeyProcessor>();

	auto typesCountsMap = getGpuTypesCountsMap(counts, keyProc.get());

	auto counter = AnyLengthInstancesUniquePrevalenceProvider(typesCountsMap);

	CliquesCandidates candidates = {
		{ 0xA, 0xB }
	};

	auto gpuCandidates = Entities::moveCliquesCandidatesToGpu(candidates);

	auto instanceTreeResult = std::make_shared<InstanceTree::InstanceTreeResult>();

	std::vector<FeatureInstance> itInstances = { 
		{ 0xA0000 },{ 0xA0000 }
		,{ 0xB0000 },{ 0xB0001 }
	};
	std::vector<unsigned int> itCliqueId = { 0, 0 };

	instanceTreeResult->instances = itInstances;
	instanceTreeResult->instancesCliqueId = itCliqueId;

	auto result = counter.getPrevalenceFromCandidatesInstances(
		gpuCandidates
		, instanceTreeResult
	);

	std::vector<float> expected = { 0.5f };

	thrust::host_vector<float> hResult = *result;
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	REQUIRE(std::equal(expected.begin(), expected.end(), hResult.begin()));
}

TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProvider | simple 2")
{
	auto counts = std::make_shared<TypesCounts>();
	counts->push_back(TypeCount(0xA, 2));
	counts->push_back(TypeCount(0xB, 4));
	counts->push_back(TypeCount(0xC, 2));

	auto keyProc = std::make_shared<GPUUIntKeyProcessor>();

	auto typesCountsMap = getGpuTypesCountsMap(counts, keyProc.get());

	auto counter = AnyLengthInstancesUniquePrevalenceProvider(typesCountsMap);

	CliquesCandidates candidates = {
		{ 0xA, 0xB }
	};

	auto gpuCandidates = Entities::moveCliquesCandidatesToGpu(candidates);

	auto instanceTreeResult = std::make_shared<InstanceTree::InstanceTreeResult>();

	std::vector<FeatureInstance> itInstances = {
		{ 0xA0000 },{ 0xA0000 }
		,{ 0xB0000 },{ 0xB0001 }
	};
	std::vector<unsigned int> itCliqueId = { 0, 0 };

	instanceTreeResult->instances = itInstances;
	instanceTreeResult->instancesCliqueId = itCliqueId;

	auto result = counter.getPrevalenceFromCandidatesInstances(
		gpuCandidates
		, instanceTreeResult
	);

	std::vector<float> expected = { 0.5f };

	thrust::host_vector<float> hResult = *result;
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	REQUIRE(std::equal(expected.begin(), expected.end(), hResult.begin()));
}


TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProvider | multiple types 0")
{
	auto counts = std::make_shared<TypesCounts>();
	counts->push_back(TypeCount(0xA, 2));
	counts->push_back(TypeCount(0xB, 4));
	counts->push_back(TypeCount(0xC, 2));

	auto keyProc = std::make_shared<GPUUIntKeyProcessor>();

	auto typesCountsMap = getGpuTypesCountsMap(counts, keyProc.get());

	auto counter = AnyLengthInstancesUniquePrevalenceProvider(typesCountsMap);

	CliquesCandidates candidates = {
		{ 0xA, 0xB }
		, { 0xA, 0xC }
		, { 0xB, 0xC }
	};

	auto gpuCandidates = Entities::moveCliquesCandidatesToGpu(candidates);

	auto instanceTreeResult = std::make_shared<InstanceTree::InstanceTreeResult>();

	std::vector<FeatureInstance> itInstances = {
		{ 0xA0000 },{ 0xA0000 },{ 0xA0000 },{ 0xB0000 }
		,{ 0xB0000 },{ 0xB0001 },{ 0xC0001 }, { 0xC0001 }
	};
	std::vector<unsigned int> itCliqueId = { 0, 0, 1, 2 };

	instanceTreeResult->instances = itInstances;
	instanceTreeResult->instancesCliqueId = itCliqueId;

	auto result = counter.getPrevalenceFromCandidatesInstances(
		gpuCandidates
		, instanceTreeResult
	);

	std::vector<float> expected = { 0.5f, 0.5f, 0.25f };

	thrust::host_vector<float> hResult = *result;
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	REQUIRE(std::equal(expected.begin(), expected.end(), hResult.begin()));
}

TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProvider | multiple types 1")
{
	auto counts = std::make_shared<TypesCounts>();
	counts->push_back(TypeCount(0xA, 2));
	counts->push_back(TypeCount(0xB, 4));
	counts->push_back(TypeCount(0xC, 2));

	auto keyProc = std::make_shared<GPUUIntKeyProcessor>();

	auto typesCountsMap = getGpuTypesCountsMap(counts, keyProc.get());

	auto counter = AnyLengthInstancesUniquePrevalenceProvider(typesCountsMap);

	CliquesCandidates candidates = {
		{ 0xA, 0xB }
		,{ 0xA, 0xC }
		,{ 0xB, 0xC }
	};

	auto gpuCandidates = Entities::moveCliquesCandidatesToGpu(candidates);

	auto instanceTreeResult = std::make_shared<InstanceTree::InstanceTreeResult>();

	std::vector<FeatureInstance> itInstances = {
		{ 0xA0000 },{ 0xA0000 },{ 0xB0000 }
		,{ 0xB0000 },{ 0xB0001 }, { 0xC0001 }
	};
	std::vector<unsigned int> itCliqueId = { 0, 0, 2 };

	instanceTreeResult->instances = itInstances;
	instanceTreeResult->instancesCliqueId = itCliqueId;

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	auto result = counter.getPrevalenceFromCandidatesInstances(
		gpuCandidates
		, instanceTreeResult
	);

	std::vector<float> expected = { 0.5f, 0.f, 0.25f };

	thrust::host_vector<float> hResult = *result;

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	REQUIRE(std::equal(expected.begin(), expected.end(), hResult.begin()));
}

TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProvider | multiple types 2")
{
	auto counts = std::make_shared<TypesCounts>();
	counts->push_back(TypeCount(0xA, 2));
	counts->push_back(TypeCount(0xB, 4));
	counts->push_back(TypeCount(0xC, 4));
	counts->push_back(TypeCount(0xD, 5));

	auto keyProc = std::make_shared<GPUUIntKeyProcessor>();

	auto typesCountsMap = getGpuTypesCountsMap(counts, keyProc.get());

	auto counter = AnyLengthInstancesUniquePrevalenceProvider(typesCountsMap);

	CliquesCandidates candidates = {
		{ 0xA, 0xB, 0xC }
		,{ 0xA, 0xC, 0xD }
		,{ 0xB, 0xC, 0xD }
	};

	auto gpuCandidates = Entities::moveCliquesCandidatesToGpu(candidates);

	auto instanceTreeResult = std::make_shared<InstanceTree::InstanceTreeResult>();

	std::vector<FeatureInstance> itInstances = {
		{ 0xA0000 },{ 0xA0000 },{ 0xA0000 },{ 0xB0000 }
		,{ 0xB0000 },{ 0xB0001 },{ 0xC0001 },{ 0xC0001 }
		,{ 0xC0000 },{ 0xC0000 },{ 0xD0001 },{ 0xD0001 }
	};
	std::vector<unsigned int> itCliqueId = { 0, 0, 1, 2 };

	instanceTreeResult->instances = itInstances;
	instanceTreeResult->instancesCliqueId = itCliqueId;

	auto result = counter.getPrevalenceFromCandidatesInstances(
		gpuCandidates
		, instanceTreeResult
	);

	std::vector<float> expected = { 0.25f, 0.2f, 0.2f };

	thrust::host_vector<float> hResult = *result;
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	REQUIRE(std::equal(expected.begin(), expected.end(), hResult.begin()));
}


TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProvider | multiple types 3")
{
	auto counts = std::make_shared<TypesCounts>();
	counts->push_back(TypeCount(0xA, 2));
	counts->push_back(TypeCount(0xB, 4));
	counts->push_back(TypeCount(0xC, 4));
	counts->push_back(TypeCount(0xD, 5));

	auto keyProc = std::make_shared<GPUUIntKeyProcessor>();

	auto typesCountsMap = getGpuTypesCountsMap(counts, keyProc.get());

	auto counter = AnyLengthInstancesUniquePrevalenceProvider(typesCountsMap);

	CliquesCandidates candidates = {
		{ 0xA, 0xB, 0xC }
		,{ 0xA, 0xC, 0xD }
		,{ 0xB, 0xC, 0xD }
	};

	auto gpuCandidates = Entities::moveCliquesCandidatesToGpu(candidates);

	auto instanceTreeResult = std::make_shared<InstanceTree::InstanceTreeResult>();

	std::vector<FeatureInstance> itInstances = {
		{ 0xA0000 },{ 0xB0000 }
		,{ 0xC0001 },{ 0xC0001 }
		,{ 0xD0001 },{ 0xD0001 }
	};
	std::vector<unsigned int> itCliqueId = {  1, 2 };

	instanceTreeResult->instances = itInstances;
	instanceTreeResult->instancesCliqueId = itCliqueId;

	auto result = counter.getPrevalenceFromCandidatesInstances(
		gpuCandidates
		, instanceTreeResult
	);

	std::vector<float> expected = { 0.0f, 0.2f, 0.2f };

	thrust::host_vector<float> hResult = *result;
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	REQUIRE(std::equal(expected.begin(), expected.end(), hResult.begin()));
}

TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProvider | multiple types 4")
{
	auto counts = std::make_shared<TypesCounts>();
	counts->push_back(TypeCount(0xA, 2));
	counts->push_back(TypeCount(0xB, 4));
	counts->push_back(TypeCount(0xC, 4));
	counts->push_back(TypeCount(0xD, 5));
	counts->push_back(TypeCount(0xE, 5));
	counts->push_back(TypeCount(0xF, 5));
	counts->push_back(TypeCount(0x10, 10));

	auto keyProc = std::make_shared<GPUUIntKeyProcessor>();

	auto typesCountsMap = getGpuTypesCountsMap(counts, keyProc.get());

	auto counter = AnyLengthInstancesUniquePrevalenceProvider(typesCountsMap);

	CliquesCandidates candidates = {
		{ 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x10 }
	};

	auto gpuCandidates = Entities::moveCliquesCandidatesToGpu(candidates);

	auto instanceTreeResult = std::make_shared<InstanceTree::InstanceTreeResult>();

	std::vector<FeatureInstance> itInstances = {
		{ 0xA0000 }
		,{ 0xB0001 }
		,{ 0xC0001 }
		,{ 0xD0001 }
		,{ 0xE0001 }
		,{ 0xF0001 }
		,{ 0x100001 }
	};
	std::vector<unsigned int> itCliqueId = { 0 };

	instanceTreeResult->instances = itInstances;
	instanceTreeResult->instancesCliqueId = itCliqueId;

	auto result = counter.getPrevalenceFromCandidatesInstances(
		gpuCandidates
		, instanceTreeResult
	);

	std::vector<float> expected = { 0.1f };

	thrust::host_vector<float> hResult = *result;
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	REQUIRE(std::equal(expected.begin(), expected.end(), hResult.begin()));
}