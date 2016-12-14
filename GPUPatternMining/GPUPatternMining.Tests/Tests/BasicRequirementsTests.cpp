#include "../catch.hpp"

#include <driver_types.h>
#include <type_traits>

#include "../../GPUPatternMining.Contract/Enity/FeatureInstance.h"

#include "../BaseCudaTestHandler.h"
//--------------------------------------------------------------

TEST_CASE("CheckIfShortIs2Bytes", "BasicRequirementsTests")
{
	REQUIRE(sizeof(short) == 2);
}

TEST_CASE("CheckIfFloatIs4Bytes", "BasicRequirementsTests")
{
	REQUIRE(sizeof(float) == 4);
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Minimum CC 2.0", "BasicRequirementsTests")
{
	int devicesCount;
	cudaDeviceProp props;

	cudaGetDeviceCount(&devicesCount);

	REQUIRE(devicesCount > 0);

	int countOfSufficientDevices = 0;

	for (int i = 0; i < devicesCount; ++i)
	{
		cudaGetDeviceProperties(&props, i);
		
		if (props.major >= 2)
			++countOfSufficientDevices;

	}

	REQUIRE(countOfSufficientDevices > 0);
}

TEST_CASE("Check if FeatureInstance struct is POD", "BasicRequirementsTests")
{
	REQUIRE(std::is_pod<FeatureInstance>::value == true);
}