#pragma once
#include <vector>
#include <memory>
#include <algorithm>

class CinsNode
{
public:
	std::vector<std::unique_ptr<CinsNode>> children;
	CinsNode* parent;
	unsigned short instanceId; //4,5,12
	unsigned short type; //A,B,C

	unsigned short addChild(unsigned short instanceId, unsigned short type);
	CinsNode* addChildPtr(unsigned short instanceId, unsigned short type);
	CinsNode* indexChild(unsigned short instanceId, unsigned short type);
	std::vector<CinsNode*> getAncestors();
	std::vector<CinsNode*> getPath();
	unsigned short getDepth();
	CinsNode();
	~CinsNode();
};

