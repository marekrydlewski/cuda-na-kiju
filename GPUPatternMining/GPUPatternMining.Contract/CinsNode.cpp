#include "CinsNode.h"

#include <vector>
#include <memory>


void CinsNode::addChild(unsigned int instanceId, unsigned int type)
{
	std::shared_ptr<CinsNode> ptr(new CinsNode());
	ptr->instanceId = instanceId;
	ptr->type = type;
	children.push_back(ptr);
}

CinsNode::CinsNode()
{
}


CinsNode::~CinsNode()
{
}
