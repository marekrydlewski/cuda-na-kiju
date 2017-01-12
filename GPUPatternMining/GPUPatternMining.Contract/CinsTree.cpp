#include "CinsTree.h"

///Root is level 0
unsigned short CinsTree::getDepth()
{
	std::vector<unsigned short> depths;
	for (auto& c : root->children)
	{
		depths.push_back(c->getDepth());
	}
	auto max = std::max_element(depths.begin(), depths.end());
	return (max != depths.end() ? *max : 0);
}

CinsTree::CinsTree()
{
	std::unique_ptr<CinsNode> ptr(new CinsNode());
	ptr->parent = nullptr;
	root = std::move(ptr);
}


CinsTree::~CinsTree()
{
}
