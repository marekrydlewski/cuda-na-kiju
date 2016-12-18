#include "CinsTree.h"

///Root is level 0
unsigned int CinsTree::getDepth()
{
	std::vector<unsigned int> depths;
	for (auto& c : root->children)
	{
		depths.push_back(c->getDepth());
	}
	auto max = std::max_element(depths.begin(), depths.end());
	return (max != depths.end() ? *max : 0);
}

CinsTree::CinsTree()
{
	root = nullptr;
}


CinsTree::~CinsTree()
{
}
