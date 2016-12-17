#include "CinsTree.h"



///Root is level 0
unsigned int CinsTree::getDepth()
{
	for (auto& c : root->children)
	{
		c->getDepth();
	}
	return 0;
}

CinsTree::CinsTree()
{
	root = nullptr;
}


CinsTree::~CinsTree()
{
}
