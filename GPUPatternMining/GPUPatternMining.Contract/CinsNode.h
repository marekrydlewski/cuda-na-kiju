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

	unsigned int addChild(unsigned int instanceId, unsigned int type);
	CinsNode* addChildPtr(unsigned int instanceId, unsigned int type);
	CinsNode* indexChild(unsigned int instanceId, unsigned int type);
	std::vector<CinsNode*> getAncestors();
	std::vector<CinsNode*> getPath();
	unsigned int getDepth();
	CinsNode();
	~CinsNode();
};

