#include "../catch.hpp"

#include <map>

#include "../../GPUPatternMining/HashMap/gpuhashmapper.h"

#include "../BaseCudaTestHandler.h"
//--------------------------------------------------------------

TEST_CASE_METHOD(BaseCudaTestHandler, "insert test", "HashMap")
{

	const int hashSize = 200;
	
	auto kp = GPUUIntKeyProcessor();
	
	auto gpuHashMapper = GPUHashMapper<unsigned int, unsigned int, GPUUIntKeyProcessor>(
		hashSize, &kp);
	
	//gpuHashMapper.clean();

	unsigned int keys[2] = {1, 2};
	unsigned int value[1] = { 3 };

	gpuHashMapper.insertKeyValuePairs(keys, value, 1);
	
}