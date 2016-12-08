#include "../catch.hpp"

#define TEST_CUDA_CHECK_RETURN
//--------------------------------------------------------------

#include <thrust/device_vector.h>

#include "../BaseCudaTestHandler.h"

#include "../../GPUPatternMining.Contract/IPairColocationsProvider.h"
#include "../../GPUPatternMining/HashMap/gpuhashmapper.h"
#include "../../GPUPatternMining/PairColocationsFiltering/CudaPairColocationFilter.h"
//--------------------------------------------------------------

/*
	Initializes data according to grap in PB.pdf on page 117

	Memory allocated on device will be released in onTeardown method by cudaDeviceReset so we don't
	have to worry about that in tests
*/
void initData(UIntTableGpuHashMapPtr& pairColocationInstancesListMap, UIntGpuHashMapPtr& pairColocationInstancesCountMap)
{
	constexpr size_t ntc = 6; // nodesTypesCount

	// hashSize = ((ntc^ntc) - ntc) / 2 + ntc = |upper right part of matrix with diagonal|
	constexpr size_t hashSize = (ntc * ntc - ntc) / 2 + ntc;

	auto intKeyProcessor = std::make_shared<GPUUIntKeyProcessor>();
	pairColocationInstancesListMap = std::make_shared<UIntTableGpuHashMap>(hashSize, intKeyProcessor.get());
	pairColocationInstancesCountMap = std::make_shared<UIntGpuHashMap>(hashSize, intKeyProcessor.get());
	
	// keys
	size_t keyTableSize = 15 * uintSize;
	unsigned int h_keys[15] = { 0xAA, 0xAB, 0xAC, 0xAD, 0xAF, 0xBB, 0xBC, 0xBD, 0xBF, 0xCC, 0xCD, 0xCF, 0xDD, 0xDF, 0xFF };
	unsigned int* c_keys;
	cudaMalloc(reinterpret_cast<void**>(&c_keys), keyTableSize);
	cudaMemcpy(c_keys, h_keys, keyTableSize, cudaMemcpyHostToDevice);

	// instances count
	size_t pairColocationsCountSize = 15 * uintSize;
	
	//values in this table are {instances count} * 2 beacuse one instance is represented as two uInts
	unsigned int h_pairColocationsInstancesCount[15] = { 0, 6, 12, 4, 6, 0, 6, 2, 2, 0, 0, 0, 0, 4, 0 };
	
	unsigned int *c_pairColocationsInstancesCount;
	cudaMalloc(reinterpret_cast<void**>(&c_pairColocationsInstancesCount), pairColocationsCountSize);
	cudaMemcpy(c_pairColocationsInstancesCount, h_pairColocationsInstancesCount, pairColocationsCountSize, cudaMemcpyHostToDevice);
	pairColocationInstancesCountMap->insertKeyValuePairs(c_keys,  c_pairColocationsInstancesCount, 15);

	// instances lists
	constexpr size_t instancesCount = 42;
	size_t pairColocationsSize = instancesCount * uintSize;
	unsigned int h_instancesLists[instancesCount] = {
		//AA
		1,2, 2,4, 3,4,  //AB
		1,1, 1,2, 2,3, 3,1, 3,2, 3,3,  //AC
		1,1, 2,2,  //AD
		1,4, 1,2, 2,2,  //AF
		//BB
		2,2, 4,2, 4,3,  //BC
		2,1,  //BD
		2,2,  //BF
		//CC
		//CD
		//CF
		//DD
		1,2, 2,2  //DF
		//FF
	};

	unsigned int* c_instancesList;
	cudaMalloc(reinterpret_cast<void**>(&c_instancesList), pairColocationsSize);
	cudaMemcpy(c_instancesList, h_instancesLists, pairColocationsSize, cudaMemcpyHostToDevice);

	size_t keyInstanceListTableSize = 15 * uintPtrSize;
	UInt* h_keyInstanceListTable[15] = {
		NULL,					//AA
		c_instancesList,		//AB
		c_instancesList + 6,	//AC
		c_instancesList + 18,	//AD
		c_instancesList + 22,	//AF
		NULL,					//BB
		c_instancesList + 28,	//BC
		c_instancesList + 34,	//BD
		c_instancesList + 36,	//BF 
		NULL,					//CC
		NULL,					//CD
		NULL,					//CF
		NULL,					//DD
		c_instancesList + 38,	//DF 
		NULL,					//FF
	};

	UInt** c_keyInstanceListTable;
	CUDA_CHECK_RETURN(cudaMalloc(reinterpret_cast<void**>(&c_keyInstanceListTable), keyInstanceListTableSize));
	CUDA_CHECK_RETURN(cudaMemcpy(c_keyInstanceListTable, h_keyInstanceListTable, keyInstanceListTableSize, cudaMemcpyHostToDevice));
	pairColocationInstancesListMap->insertKeyValuePairs(c_keys, c_keyInstanceListTable, 15);
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Simple initial data test 00", "Prevalence index test data test")
{
	UIntTableGpuHashMapPtr colocationInstancesListMap;
	UIntGpuHashMapPtr colocationInstancesCountMap;

	initData(colocationInstancesListMap, colocationInstancesCountMap);

	unsigned int h_key[] = { 0xCC };
	unsigned int* c_key;
	cudaMalloc(reinterpret_cast<void**>(&c_key), uintSize);
	cudaMemcpy(c_key, h_key, uintSize, cudaMemcpyHostToDevice);

	unsigned int h_instanceCount;
	unsigned int* c_instancesCount;
	cudaMalloc(reinterpret_cast<void**>(&c_instancesCount), uintSize);

	colocationInstancesCountMap->getValues(c_key, c_instancesCount, 1);
	cudaMemcpy(&h_instanceCount, c_instancesCount, uintSize, cudaMemcpyDeviceToHost);

	REQUIRE(h_instanceCount == 0);

	unsigned int** c_instancesListPtr;
	cudaMalloc(reinterpret_cast<void**>(&c_instancesListPtr), uintPtrSize);
	colocationInstancesListMap->getValues(c_key, c_instancesListPtr, 1);

	unsigned int* h_instanceListPtr;
	cudaMemcpy(&h_instanceListPtr, c_instancesListPtr, uintPtrSize, cudaMemcpyDeviceToHost);

	REQUIRE(h_instanceListPtr == NULL);
}


TEST_CASE_METHOD(BaseCudaTestHandler, "Simple initial data test 01", "Prevalence index test data test")
{
	UIntTableGpuHashMapPtr colocationInstancesListMap;
	UIntGpuHashMapPtr colocationInstancesCountMap;

	initData(colocationInstancesListMap, colocationInstancesCountMap);

	unsigned int h_key[] = { 0xAC };
	unsigned int* c_key;
	cudaMalloc(reinterpret_cast<void**>(&c_key), uintSize);
	cudaMemcpy(c_key, h_key, uintSize, cudaMemcpyHostToDevice);

	unsigned int h_instanceCount;
	unsigned int* c_instancesCount;
	cudaMalloc(reinterpret_cast<void**>(&c_instancesCount), uintSize);
		
	colocationInstancesCountMap->getValues(c_key, c_instancesCount, 1);	
	cudaMemcpy(&h_instanceCount, c_instancesCount, uintSize, cudaMemcpyDeviceToHost);

	REQUIRE(h_instanceCount == 6 * 2);

	std::shared_ptr<unsigned int> h_instances(new unsigned int[h_instanceCount], std::default_delete<unsigned int[]>());

	unsigned int** c_instancesListPtr;
	cudaMalloc(reinterpret_cast<void**>(&c_instancesListPtr), uintPtrSize);
	colocationInstancesListMap->getValues(c_key, c_instancesListPtr, 1);

	unsigned int* h_instanceListPtr;
	cudaMemcpy(&h_instanceListPtr, c_instancesListPtr, uintPtrSize, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_instances.get(), h_instanceListPtr, h_instanceCount * uintSize, cudaMemcpyDeviceToHost);

	unsigned int expected[] = { 1,1, 1,2, 2,3, 3,1, 3,2, 3,3 };

	REQUIRE(std::equal(std::begin(expected), std::end(expected), h_instances.get()) == true);
}


TEST_CASE_METHOD(BaseCudaTestHandler, "Prevalence index test 01", "HashMap")
{
	UIntTableGpuHashMapPtr colocationInstancesListMap;
	UIntGpuHashMapPtr colocationInstancesCountMap;

	initData(colocationInstancesListMap, colocationInstancesCountMap);

	
	
	REQUIRE(true);
}
