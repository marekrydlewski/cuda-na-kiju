#include "catch.hpp"

#include <map> 
#include <thrust/device_vector.h>
#include <thrust/unique.h>
//----------------------------------------------------------------------------------------


#define TEST_CUDA_CHECK_RETURN
//----------------------------------------------------------------------------------------

#include "BaseCudaTestHandler.h"
//----------------------------------------------------------------------------------------


TEST_CASE_METHOD(BaseCudaTestHandler, "AnyLengthInstancesUniquePrevalenceProvider | simple")
{

}