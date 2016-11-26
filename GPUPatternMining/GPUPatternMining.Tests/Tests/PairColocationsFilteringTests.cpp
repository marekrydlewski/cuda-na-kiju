#include "../catch.hpp"


#include "../BaseCudaTestHandler.h"

#include "../../GPUPatternMining.Contract/IPairColocationsProvider.h"
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

	void* getPairColocations() override
	{
		return data;
	}

	int getPairColocationsCount() override
	{
		return 0;
	}

private:

	void* data;
};
//--------------------------------------------------------------


TEST_CASE_METHOD(BaseCudaTestHandler, "CheckIfZeroIsNotOne", "PairColocationsFilteringTests")
{
	REQUIRE(1 != 0);
}