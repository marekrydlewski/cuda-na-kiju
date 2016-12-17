#pragma once

#include<memory>
#include "CinsNode.h"

class CinsTree
{
public:
	std::unique_ptr<CinsNode> root;

	unsigned int getDepth();
	CinsTree();
	~CinsTree();
};

