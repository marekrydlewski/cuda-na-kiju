#include "CinsNode.h"

#include <vector>
#include <memory>


unsigned short CinsNode::addChild(unsigned short instanceId, unsigned short type)
{
	std::unique_ptr<CinsNode> ptr(new CinsNode());
	ptr->instanceId = instanceId;
	ptr->type = type;
	ptr->parent = this;

	size_t index = children.size();
	children.push_back(std::move(ptr));
	return index;
}

CinsNode * CinsNode::addChildPtr(unsigned short instanceId, unsigned short type)
{
	std::unique_ptr<CinsNode> ptr(new CinsNode());
	ptr->instanceId = instanceId;
	ptr->type = type;
	ptr->parent = this;

	children.push_back(std::move(ptr));
	return children.back().get();
}


CinsNode* CinsNode::indexChild(unsigned short instanceId, unsigned short type)
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
	node = node->parent;
	while (node->parent != nullptr)
	{
		ancestors.push_back(node);
		node = node->parent;
	}
	return ancestors;
}

std::vector<CinsNode*> CinsNode::getPath()
{
	CinsNode* node = this;
	std::vector<CinsNode*> ancestors;
	while (node->parent != nullptr)
	{
		ancestors.push_back(node);
		node = node->parent;
	}
	return ancestors;
}

unsigned short CinsNode::getDepth()
{
	std::vector<unsigned short> depths;
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
