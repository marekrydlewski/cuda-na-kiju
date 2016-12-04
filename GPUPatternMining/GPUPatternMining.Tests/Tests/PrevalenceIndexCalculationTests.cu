#include "../catch.hpp"

#include <thrust/device_vector.h>

#include "../BaseCudaTestHandler.h"

#include "../../GPUPatternMining.Contract/IPairColocationsProvider.h"
#include "../../GPUPatternMining/HashMap/gpuhashmapper.h"
#include "../../GPUPatternMining/PairColocationsFiltering/CudaPairColocationFilter.h"
//--------------------------------------------------------------


typedef thrust::host_vector<unsigned int> HostUIntVector;
typedef thrust::device_vector<unsigned int> DeviceUIntVector;
typedef std::vector<unsigned int > UIntVector;
//--------------------------------------------------------------

/*
	Initializes data according to grap in PB.pdf on page 117

	Memory allocated on device will be released in onTeardown method by cudaDeviceReset so we don't
	have to worry about that in tests
*/
void initData(UIntTableGpuHashMapPtr& colocationInstancesListMap, UIntGpuHashMapPtr& colocationInstancesCountMap)
{
	constexpr size_t ntc = 6; // nodesTypesCount

	// hashSize = ((ntc^ntc) - ntc) / 2 + ntc = |upper right part of matrix with diagonal|
	constexpr size_t hashSize = (ntc * ntc - ntc) / 2 + ntc;

	auto intKeyProcessor = std::make_shared<GPUUIntKeyProcessor>();
	colocationInstancesListMap = std::make_shared<UIntTableGpuHashMap>(hashSize, intKeyProcessor.get());
	colocationInstancesCountMap = std::make_shared<UIntGpuHashMap>(hashSize, intKeyProcessor.get());
	
	// keys
	size_t keyTableSize = 15 * sizeof(unsigned int);
	unsigned int h_keys[15] = { 0xAA, 0xAB, 0xAC, 0xAD, 0xAF, 0xBB, 0xBC, 0xBD, 0xBF, 0xCC, 0xCD, 0xCF, 0xDD, 0xDF, 0xFF };
	unsigned int* c_keys;
	cudaMalloc(reinterpret_cast<void**>(&c_keys), keyTableSize);
	cudaMemcpy(c_keys, h_keys, keyTableSize, cudaMemcpyHostToDevice);

	// instances count
	size_t neighboursCountSize = 15 * sizeof(unsigned int);
	
	//values in this table are {instances count} * 2 beacuse of one instance is represented as two uInts
	unsigned int h_pairColocationsInstancesCount[15] = { 0, 6, 12, 4, 6, 0, 6, 2, 2, 0, 0, 0, 0, 4, 0 };
	
	unsigned int *c_pairColocationsInstancesCount;
	cudaMalloc(reinterpret_cast<void**>(&c_pairColocationsInstancesCount), neighboursCountSize);
	cudaMemcpy(c_pairColocationsInstancesCount, h_pairColocationsInstancesCount, neighboursCountSize, cudaMemcpyHostToDevice);
	colocationInstancesCountMap->insertKeyValuePairs(c_keys,  c_pairColocationsInstancesCount, 15);

	// instances lists
	constexpr size_t instancesCount = 42;
	size_t pairColocationsSize = instancesCount * sizeof(unsigned int);
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

	size_t keyInstanceListTableSize = 15 * sizeof(unsigned int*);
	unsigned int* h_keyInstanceListTable[15] = {
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

	unsigned int** c_keyInstanceListTable;
	cudaMalloc(reinterpret_cast<void**>(&c_keyInstanceListTable), keyInstanceListTableSize);
	cudaMemcpy(c_keyInstanceListTable, h_keyInstanceListTable, keyInstanceListTableSize, cudaMemcpyHostToDevice);
	colocationInstancesListMap->insertKeyValuePairs(c_keys, c_keyInstanceListTable, 15);
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Simple initial data test 00", "Prevalence index test data test")
{
	UIntTableGpuHashMapPtr colocationInstancesListMap;
	UIntGpuHashMapPtr colocationInstancesCountMap;

	initData(colocationInstancesListMap, colocationInstancesCountMap);

	unsigned int h_key[] = { 0xCC };
	unsigned int* c_key;
	cudaMalloc(reinterpret_cast<void**>(&c_key), sizeof(unsigned int));
	cudaMemcpy(c_key, h_key, sizeof(unsigned int), cudaMemcpyHostToDevice);

	unsigned int h_instanceCount;
	unsigned int* c_instancesCount;
	cudaMalloc(reinterpret_cast<void**>(&c_instancesCount), sizeof(unsigned int));

	colocationInstancesCountMap->getValues(c_key, c_instancesCount, 1);
	cudaMemcpy(&h_instanceCount, c_instancesCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	REQUIRE(h_instanceCount == 0);

	unsigned int** c_instancesListPtr;
	cudaMalloc(reinterpret_cast<void**>(&c_instancesListPtr), sizeof(unsigned int*));
	colocationInstancesListMap->getValues(c_key, c_instancesListPtr, 1);

	REQUIRE(c_instancesListPtr == NULL);
}


TEST_CASE_METHOD(BaseCudaTestHandler, "Simple initial data test 01", "Prevalence index test data test")
{
	UIntTableGpuHashMapPtr colocationInstancesListMap;
	UIntGpuHashMapPtr colocationInstancesCountMap;

	initData(colocationInstancesListMap, colocationInstancesCountMap);

	unsigned int h_key[] = { 0xAC };
	unsigned int* c_key;
	cudaMalloc(reinterpret_cast<void**>(&c_key), sizeof(unsigned int));
	cudaMemcpy(c_key, h_key, sizeof(unsigned int), cudaMemcpyHostToDevice);

	unsigned int h_instanceCount;
	unsigned int* c_instancesCount;
	cudaMalloc(reinterpret_cast<void**>(&c_instancesCount), sizeof(unsigned int));
		
	colocationInstancesCountMap->getValues(c_key, c_instancesCount, 1);	
	cudaMemcpy(&h_instanceCount, c_instancesCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	REQUIRE(h_instanceCount == 6 * 2);

	//std::shared_ptr<unsigned int> h_instances(new unsigned int[h_instanceCount], std::default_delete<unsigned int[]>());
	unsigned int* h_instances = new unsigned int[h_instanceCount];

	unsigned int** c_instancesListPtr;
	cudaMalloc(reinterpret_cast<void**>(&c_instancesListPtr), sizeof(unsigned int*));
	colocationInstancesListMap->getValues(c_key, c_instancesListPtr, 1);

	cudaMemcpy(h_instances, c_instancesListPtr, h_instanceCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	unsigned int expected[] = { 1,1, 1,2, 2,3, 3,1, 3,2, 3,3 };

	REQUIRE(std::equal(std::begin(expected), std::end(expected), h_instances) == true);
}


TEST_CASE_METHOD(BaseCudaTestHandler, "Prevalence index test 01", "HashMap")
{
	UIntTableGpuHashMapPtr colocationInstancesListMap;
	UIntGpuHashMapPtr colocationInstancesCountMap;

	initData(colocationInstancesListMap, colocationInstancesCountMap);

	
	
	REQUIRE(true);
}
