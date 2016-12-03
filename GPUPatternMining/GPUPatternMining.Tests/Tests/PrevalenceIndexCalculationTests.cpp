#include "../catch.hpp"


#include "../BaseCudaTestHandler.h"

#include "../../GPUPatternMining.Contract/IPairColocationsProvider.h"
#include "../../GPUPatternMining/HashMap/gpuhashmapper.h"

//--------------------------------------------------------------


class SimpleMockPairProvider : public IPairColocationsProvider
{
public:

	SimpleMockPairProvider()
	{
	
	}

	~SimpleMockPairProvider()
	{
		
	}

	PairColocation* getPairColocations() override
	{
		return data;
	}

	int getPairColocationsCount() override
	{
		return 0;
	}

private:

	PairColocation* data;
};
//--------------------------------------------------------------



TEST_CASE_METHOD(BaseCudaTestHandler, "PrevalenceIndexTest_01", "HashMap")
{
	// graph used in this tests can be found in PB.pdf on page 117

	constexpr size_t maxNeighboursCount = 5;
	constexpr size_t nodesCount = 12;

	// hashSize = ((nodesCount^nodesCount) - nodesCount) / 2 = |upper right part of matrix|
	constexpr size_t hashSize = nodesCount;

	GPUUIntKeyProcessor *intKeyProcessor = new GPUUIntKeyProcessor();
	GPUHashMapper<unsigned int, unsigned int*, GPUUIntKeyProcessor> mapper(hashSize, intKeyProcessor);
	mapper.setKeyProcessor(intKeyProcessor);


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