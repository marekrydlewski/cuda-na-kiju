#pragma once
#include <vector>
#include <memory>
#include <algorithm>

class CinsNode
{
public:
	std::vector<std::unique_ptr<CinsNode>> children;
	CinsNode* parent;
	unsigned int instanceId; //4,5,12
	unsigned int type; //A,B,C

	void addChild(unsigned int instanceId, unsigned int type);
	unsigned int getDepth();
	CinsNode();
	~CinsNode();
};

