#pragma once
#include <vector>
#include <memory>

class CinsNode
{
public:
	std::vector<std::shared_ptr<CinsNode>> children;
	std::shared_ptr<CinsNode> parent;
	unsigned int instanceId; //4,5,12
	unsigned int type; //A,B,C

	void addChild(unsigned int instanceId, unsigned int type);
	CinsNode();
	~CinsNode();
};

