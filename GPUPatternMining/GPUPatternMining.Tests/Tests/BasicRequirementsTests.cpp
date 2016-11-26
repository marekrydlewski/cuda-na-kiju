#include "../catch.hpp"


#include "../BaseCudaTestHandler.h"
#include "../../GPUPatternMining/SimpleOperations.h"
//--------------------------------------------------------------

TEST_CASE("CheckIfShortIs2Bytes", "BasicRequirementsTests")
{
	REQUIRE(sizeof(short) == 2);
}

TEST_CASE("CheckIfFloatIs4Bytes", "BasicRequirementsTests")
{
	REQUIRE(sizeof(float) == 4);
}