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


TEST_CASE_METHOD(BaseCudaTestHandler, "CheckIfZeroIsNotOne", "PairColocationsFilteringTests")
{
	REQUIRE(1 != 0);
}