#include "CinsNode.h"

#include <vector>
#include <memory>


void CinsNode::addChild(unsigned int instanceId, unsigned int type)
{
	std::unique_ptr<CinsNode> ptr(new CinsNode());
	ptr->instanceId = instanceId;
	ptr->type = type;
	ptr->parent = this;
	children.push_back(ptr);
}

unsigned int CinsNode::getDepth()
{
	for (auto& c : children)
	{
		return c->getDepth() + 1;
	}
	return 1;
}

CinsNode::CinsNode()
{
}


CinsNode::~CinsNode()
{
}
