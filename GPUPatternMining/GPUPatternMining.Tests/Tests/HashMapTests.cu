#include "../catch.hpp"

#include <map>

#include "../../GPUPatternMining/HashMap/gpuhashmapper.h"

#include "../BaseCudaTestHandler.h"
//--------------------------------------------------------------

TEST_CASE_METHOD(BaseCudaTestHandler, "InsertTest", "HashMap")
{
	constexpr size_t threeUINTsize = sizeof(unsigned int) * 3;

	GPUUIntKeyProcessor *intKeyProcessor = new GPUUIntKeyProcessor();
	unsigned int hashSize = 4;

	GPUHashMapper<unsigned int, unsigned int, GPUUIntKeyProcessor> mapper(hashSize, intKeyProcessor);
	mapper.setKeyProcessor(intKeyProcessor);

	unsigned int* c_keys;
	unsigned int* c_values;

	cudaMalloc((void**)&c_keys, (sizeof(unsigned int) * 3));
	cudaMalloc((void**)&c_values, (sizeof(unsigned int) * 3));

	unsigned int h_keys[] = { 1, 2, 3 };
	unsigned int h_values[] = { 10, 100, 1000 };
	
	cudaMemcpy(c_keys, h_keys, threeUINTsize, cudaMemcpyHostToDevice);
	cudaMemcpy(c_values, h_values, threeUINTsize, cudaMemcpyHostToDevice);

	mapper.insertKeyValuePairs(c_keys, c_values, 3);

	cudaFree(c_keys);
	cudaFree(c_values);

	REQUIRE(true);
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Insert and Read test", "HashMap")
{
	constexpr size_t threeUINTsize = sizeof(unsigned int) * 3;

	GPUUIntKeyProcessor *intKeyProcessor = new GPUUIntKeyProcessor();
	unsigned int hashSize = 4;

	GPUHashMapper<unsigned int, unsigned int, GPUUIntKeyProcessor> mapper(hashSize, intKeyProcessor);
	mapper.setKeyProcessor(intKeyProcessor);

	unsigned int* c_keys;
	unsigned int* c_values;

	cudaMalloc((void**)&c_keys, (sizeof(unsigned int) * 3));
	cudaMalloc((void**)&c_values, (sizeof(unsigned int) * 3));

	unsigned int h_keys[] = { 1, 2, 3 };
	unsigned int h_values[] = { 10, 100, 1000 };

	cudaMemcpy(c_keys, h_keys, threeUINTsize, cudaMemcpyHostToDevice);
	cudaMemcpy(c_values, h_values, threeUINTsize, cudaMemcpyHostToDevice);

	mapper.insertKeyValuePairs(c_keys, c_values, 3);

	unsigned int* c_resultValues;

	cudaMalloc((void**)&c_resultValues, (sizeof(unsigned int) * 3));

	unsigned int h_resultValues[] = { 0, 0, 0 };

	mapper.getValues(c_keys, c_resultValues, 3);

	cudaMemcpy(h_resultValues, c_resultValues, threeUINTsize, cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();

	REQUIRE(h_resultValues[0] == h_values[0]);
	REQUIRE(h_resultValues[1] == h_values[1]); 
	REQUIRE(h_resultValues[2] == h_values[2]);

	cudaFree(c_keys);
	cudaFree(c_values);
	cudaFree(c_resultValues);

	REQUIRE(true);
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Insert and Read test with HEX key", "HashMap")
{
	constexpr size_t threeUINTsize = sizeof(unsigned int) * 3;

	GPUUIntKeyProcessor *intKeyProcessor = new GPUUIntKeyProcessor();
	unsigned int hashSize = 4;

	GPUHashMapper<unsigned int, unsigned int, GPUUIntKeyProcessor> mapper(hashSize, intKeyProcessor);
	mapper.setKeyProcessor(intKeyProcessor);

	unsigned int* c_keys;
	unsigned int* c_values;

	cudaMalloc((void**)&c_keys, (sizeof(unsigned int) * 3));
	cudaMalloc((void**)&c_values, (sizeof(unsigned int) * 3));

	unsigned int h_keys[] = { 0xAA, 0xAB, 0xFF };
	unsigned int h_values[] = { 10, 100, 1000 };

	cudaMemcpy(c_keys, h_keys, threeUINTsize, cudaMemcpyHostToDevice);
	cudaMemcpy(c_values, h_values, threeUINTsize, cudaMemcpyHostToDevice);

	mapper.insertKeyValuePairs(c_keys, c_values, 3);

	unsigned int* c_resultValues;

	cudaMalloc((void**)&c_resultValues, (sizeof(unsigned int) * 3));

	unsigned int h_resultValues[] = { 0, 0, 0 };

	mapper.getValues(c_keys, c_resultValues, 3);

	cudaMemcpy(h_resultValues, c_resultValues, threeUINTsize, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	REQUIRE(h_resultValues[0] == h_values[0]);
	REQUIRE(h_resultValues[1] == h_values[1]);
	REQUIRE(h_resultValues[2] == h_values[2]);

	cudaFree(c_keys);
	cudaFree(c_values);
	cudaFree(c_resultValues);

	REQUIRE(true);
}