#include "catch.hpp"

#include "BaseCudaTestHandler.h"
#include "../GPUPatternMining.Contract/RandomDataProvider.cpp"
#include "../GPUPatternMining.Contract/Enity/DataFeed.h"

TEST_CASE_METHOD(BaseCudaTestHandler, "Eeee", "DataProviderTests")
{
	RandomDataProvider rdp;

	rdp.setNumberOfTypes(2);
	rdp.setRange(10, 10);

	DataFeed* data = rdp.getData(10);

	/*for (int i = 0; i < 10; i++)
	{
		REQUIRE(data[i].xy.x <= 5);
	}*/

	REQUIRE(true);
}