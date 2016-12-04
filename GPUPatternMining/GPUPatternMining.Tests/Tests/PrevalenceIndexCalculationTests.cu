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
TEST_CASE_METHOD(BaseCudaTestHandler, "Thrust convertion by constructor test", "HashMap")
{
	auto intKeyProcessor = std::make_shared<GPUUIntKeyProcessor>();
	UIntGpuHashMapPtr container = std::make_shared<UIntGpuHashMap>(5, intKeyProcessor.get());

	HostUIntVector h_vec = HostUIntVector(3, 99);

	unsigned int* c_key;
	cudaMalloc(reinterpret_cast<void**>(&c_key), sizeof(unsigned int));

	unsigned int h_key[] = { 99 };

	cudaMemcpy(c_key, h_key, sizeof(unsigned int), cudaMemcpyHostToDevice);

	ThrustUIntVector h_value[] = { h_vec };
	ThrustUIntVector* c_value;
	cudaMalloc(reinterpret_cast<void**>(&c_value), sizeof(ThrustUIntVector));
	cudaMemcpy(c_value, h_value, sizeof(unsigned int), cudaMemcpyHostToDevice);

	container->insertKeyValuePairs(c_key, );
}
*/

void initData(UIntGpuHashMapPtr& colocationInstancesListMap, UIntGpuHashMapPtr& colocationInstancesCountMap)
{
	constexpr size_t ntc = 6; // nodesTypesCount

	// hashSize = ((ntc^ntc) - ntc) / 2 + ntc = |upper right part of matrix with diagonal|
	constexpr size_t hashSize = (ntc * ntc - ntc) / 2 + ntc;

	auto intKeyProcessor = std::make_shared<GPUUIntKeyProcessor>();
	colocationInstancesListMap = std::make_shared<UIntGpuHashMap>(hashSize, intKeyProcessor.get());
	colocationInstancesCountMap = std::make_shared<UIntGpuHashMap>(hashSize, intKeyProcessor.get());

	UIntVector AA = {};
	UIntVector AB = { 1,2, 2,4 };
	UIntVector AC = { 1,1, 1,2, 2,3, 3,1, 3,2, 3,3 };
	UIntVector AD = { 1,1, 2,2 };
	UIntVector AF = { 1,4, 1,2, 2,2 };

	UIntVector BB = {};
	UIntVector BC = { 2,2, 4,2, 4,3 };
	UIntVector BD = { 2,1 };
	UIntVector BF = { 2,2 };

	UIntVector CC = {};
	UIntVector CD = {};
	UIntVector CF = {};

	UIntVector DD = {};
	UIntVector DF = { 1,2, 2,2 };

	UIntVector FF = {};

	size_t neighboursCountSize = 15 * sizeof(unsigned int);
	unsigned int h_neighboursCount[15] = { 0, 2, 6, 2, 3, 0, 3, 1, 1, 0, 0, 0, 0, 2, 0 };
	unsigned int *c_neighboursCount;

	cudaMalloc(reinterpret_cast<void**>(&c_neighboursCount), neighboursCountSize);
	cudaMemcpy(c_neighboursCount, h_neighboursCount, neighboursCountSize, cudaMemcpyHostToDevice);
}
/*
TEST_CASE_METHOD(BaseCudaTestHandler, "Prevalence index test 01", "HashMap")
{
	// graph used in this tests can be found in PB.pdf on page 117

	constexpr size_t maxNeighboursCount = 5;
	
	UIntGpuHashMapPtr colocationInstancesListMap;
	UIntGpuHashMapPtr colocationInstancesCountMap;

	initData(colocationInstancesListMap, colocationInstancesCountMap);


	//unsigned int* c_pairs;
	//cudaMalloc(reinterpret_cast<void**>(&c_pairs), 40 * sizeof(unsigned int));
	//cudaMemcpy(c_pairs, h_pairs, 40 * sizeof(unsigned int), cudaMemcpyHostToDevice);


	
	// TODO wielkosc listy 2 * maxNeighbours * instancesOfType


	//constexpr size_t threeUINTsize = sizeof(unsigned int) * 3;
	//cudaMalloc(reinterpret_cast<void**>(&c_keys), (sizeof(unsigned int) * 3));
	//cudaMalloc(reinterpret_cast<void**>(&c_values), (sizeof(unsigned int) * 3));
	unsigned int* c_keys;
	unsigned int* c_values;

	

	unsigned int h_keys[] = { 1, 2, 3 };
	unsigned int h_values[] = { 10, 100, 1000 };

	cudaMemcpy(c_keys, h_keys, threeUINTsize, cudaMemcpyHostToDevice);
	cudaMemcpy(c_values, h_values, threeUINTsize, cudaMemcpyHostToDevice);

	mapper.insertKeyValuePairs(c_keys, c_values, 3);

	cudaFree(c_keys);
	cudaFree(c_values);
	
	REQUIRE(true);
}
*/