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
	///6 vertices, 3 types, 3 instances of t0, 2 of t1, 1 of t2, nieghbourhoods: t0i0|t1i0, t0i0|t1i2, t1i0|t2i0
	///(note duplicate neighbourhood of t0i0 with t1 vertices)
	DataFeed* getDataForMixedPrevalenceResults();
	///2 vertices, 2 types, neighbourhood relationship
	DataFeed* getDataForMaximalCliqueSize2();
	///2 vertices, 2 types, no neighbourhood relationship
	DataFeed* getDataForMaximalCliqueSize1();
	///4 vertices, 4 types, neighbourhood relationship between everyone
	DataFeed* getDataForMaximalCliqueSize4();
	///5 vertices, 5 types, hourglass-like shape where middle point belongs to both cliques
	DataFeed* getDataFor2OverlappingCliques();
	///5 vertices, 5 types, cliques of (0,1) and (2,3,4)
	DataFeed* getDataForDistinctCliques();
	///14 vertices, 5 types, cliques of (0,1,2,3) and (0,2,4) where second one is non-prevalent
	///which should result in colocations of (0,1,2,3), (0,2), (0,4), (2,4)
	DataFeed * getDataForTreeTest();
	TestDataGenerator();
	~TestDataGenerator();
};