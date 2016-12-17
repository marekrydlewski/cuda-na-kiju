#include "CinsNode.h"

#include <vector>
#include <memory>


void CinsNode::addChild(unsigned int instanceId, unsigned int type)
{
	std::unique_ptr<CinsNode> ptr(new CinsNode());
	ptr->instanceId = instanceId;
	ptr->type = type;
	ptr->parent = this;
	children.push_back(std::move(ptr));
}

unsigned int CinsNode::getDepth()
{
	std::vector<unsigned int> depths;
	for (auto& c : children)
	{
		depths.push_back(c->getDepth() + 1);
	}
	auto max = std::max_element(depths.begin(), depths.end());
	return (max != depths.end() ? *max : 1);
}

CinsNode::CinsNode()
{
}


CinsNode::~CinsNode()
{
}
