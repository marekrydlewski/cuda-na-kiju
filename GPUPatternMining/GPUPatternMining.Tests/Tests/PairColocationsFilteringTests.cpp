#include "../catch.hpp"


#include "../BaseCudaTestHandler.h"
//--------------------------------------------------------------


TEST_CASE_METHOD(BaseCudaTestHandler, "CheckIfZeroIsNotOne", "PairColocationsFilteringTests")
{
	REQUIRE(1 != 0);
}