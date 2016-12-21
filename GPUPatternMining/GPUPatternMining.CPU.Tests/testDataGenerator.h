#pragma once
#include "../GPUPatternMining.Contract/Enity/DataFeed.h"

///All generated data expects tests using neighbourhood threshold 5
class TestDataGenerator {
public:
	///5 vertices, 5 types, with huge distance between each other
	DataFeed* getNoNeighboursData();
	///5 vertices, 5 types, huge distances except types 0 and 1
	DataFeed* getOneNeighbourRelationshipData();
	///5 vertices, 5 types, linear positioning (type 0 in neighbourhood with type1, type1 with type2 and so on...)
	DataFeed* getLinearNeighbourRelationshipData();
	///6 vertices, 2 types, 4 instances of type0 and 2 instances of type1, intances 0 and 1 of both types
	///in neighbourhood (t0i0 with t1i0 and t0i1 with t1i1)
	DataFeed* getDataForPrevalenceTests();

	DataFeed* getDataForMaximalCliqueSize2();

	DataFeed* getDataForMaximalCliqueSize1();

	DataFeed* getDataForMaximalCliqueSize4();
	TestDataGenerator();
	~TestDataGenerator();
};