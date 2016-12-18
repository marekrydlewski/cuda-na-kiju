#include "CinsNode.h"

#include <vector>
#include <memory>


unsigned int CinsNode::addChild(unsigned int instanceId, unsigned int type)
{
	std::unique_ptr<CinsNode> ptr(new CinsNode());
	ptr->instanceId = instanceId;
	ptr->type = type;
	ptr->parent = this;

	size_t index = children.size();
	children.push_back(std::move(ptr));
	return index;
}

CinsNode * CinsNode::addChildPtr(unsigned int instanceId, unsigned int type)
{
	std::unique_ptr<CinsNode> ptr(new CinsNode());
	ptr->instanceId = instanceId;
	ptr->type = type;
	ptr->parent = this;

	children.push_back(std::move(ptr));
	return children.back().get();
}


CinsNode* CinsNode::indexChild(unsigned int instanceId, unsigned int type)
{
	for (auto& u : children)
	{
		if (u->instanceId == instanceId && u->type == type)
			return u.get();
	}
	return nullptr;
}

std::vector<CinsNode*> CinsNode::getAncestors()
{
	CinsNode* node = this;
	std::vector<CinsNode*> ancestors;
	while (node->parent != nullptr)
	{
		ancestors.push_back(node->parent);
		node = node->parent;
	}
	return ancestors;
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
