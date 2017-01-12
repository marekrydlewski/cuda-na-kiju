#pragma once

#include<memory>
#include "CinsNode.h"

class CinsTree
{
public:
	std::unique_ptr<CinsNode> root;
	std::vector<CinsNode*> lastLevelChildren;

	unsigned short getDepth();
	CinsTree();
	~CinsTree();
};

