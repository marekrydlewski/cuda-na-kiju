#include "../catch.hpp"


#include "../BaseCudaTestHandler.h"
#include "../../GPUPatternMining/HashMap/gpuhashmapper.h"
//--------------------------------------------------------------


TEST_CASE_METHOD(BaseCudaTestHandler, "HashMapBasicTestInsertValues", "HashMapTests")
{
	GPUUIntKeyProcessor *intKeyProcessor = new GPUUIntKeyProcessor();
	unsigned int hashSize = 4;

	GPUHashMapper<unsigned int, unsigned int, GPUUIntKeyProcessor> mapper(hashSize, intKeyProcessor);
	mapper.setKeyProcessor(intKeyProcessor);

	unsigned int keys[] = { 1, 2, 3 }, values[] = { 10, 100, 1000 };

	mapper.insertKeyValuePairs(keys, values, 3);

	REQUIRE(true);
}
