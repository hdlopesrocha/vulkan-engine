#include "OctreeNode.hpp"
#include "OctreeNodeCubeSerialized.hpp"
#include "OctreeAllocator.hpp"
#include "OctreeNodeData.hpp"
#include "OctreeChangeHandler.hpp"
#include <cmath>
#include <cstring>


const float INFINITY_ARRAY [8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};


OctreeNode::OctreeNode() {
	version = 0u;
	init(glm::vec3());
}

OctreeNode::OctreeNode(Vertex vertex_) {
	version = 0u;
	init(vertex_);
}

OctreeNode * OctreeNode::init(Vertex vert) {
	memcpy(this->sdf, INFINITY_ARRAY, sizeof(float)*8);
	this->bits = 0x0;
	this->setSimplification(0u);
	this->setChunk(false);
	this->setType(SpaceType::Surface);
	this->vertex = vert;
	this->blockId = UINT_MAX;
	this->version = 0u;	
	return this;
}

ChildBlock * OctreeNode::getBlock(OctreeAllocator &allocator) const {
	return allocator.childAllocator.getFromIndex(this->blockId);
}


void OctreeNode::setChildren(OctreeAllocator &allocator, OctreeNode * childrenPtr[8]) {
	uint localBlockId = this->blockId;
	ChildBlock * block = NULL;
	if(localBlockId == UINT_MAX) {
		block = allocator.childAllocator.allocate()->init();
		this->blockId = allocator.childAllocator.getIndex(block);
	} else {
		block = allocator.childAllocator.getFromIndex(localBlockId);
	}

    uint childNodes[8] = {UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX};
	for(int i=0; i < 8; ++i) {
		if(childrenPtr[i] != NULL) {
			childNodes[i] = allocator.getIndex(childrenPtr[i]);
		}
	}
	memcpy(block->children, childNodes, sizeof(uint)*8);
}

void OctreeNode::setChildren(OctreeAllocator &allocator, uint children[8]) {
	uint localBlockId = this->blockId;
	ChildBlock * block = NULL;
	if(localBlockId == UINT_MAX) {
		block = allocator.childAllocator.allocate()->init();
		this->blockId = allocator.childAllocator.getIndex(block);
	} else {
		block = allocator.childAllocator.getFromIndex(localBlockId);
	}
	memcpy(block->children, children, sizeof(uint)*8);
}

void OctreeNode::getChildren(OctreeAllocator &allocator, OctreeNode * childNodes[8]) const {
	ChildBlock * block = getBlock(allocator);
	if(block != NULL) {
	    allocator.get(childNodes, block->children);
	}
}

ChildBlock * OctreeNode::allocate(OctreeAllocator &allocator) {
	ChildBlock * block = NULL;
	if(this->blockId == UINT_MAX) {
		block = allocator.childAllocator.allocate();
		this->blockId = allocator.childAllocator.getIndex(block);
	}
	if(block == NULL) {
		block = allocator.childAllocator.getFromIndex(this->blockId);
	}
	return block;
}



OctreeNode::~OctreeNode() {

}

ChildBlock * OctreeNode::clear(OctreeAllocator &allocator, ChildBlock * block) {

	if(this->blockId != UINT_MAX) {
		if(block == NULL) {
			block = getBlock(allocator);
		}
		if(block!=NULL) {	
			block->clear(allocator);
			allocator.childAllocator.deallocate(block);
			this->blockId = UINT_MAX;
			block = NULL;
		}
	}
	return block;
}

void OctreeNode::setSDF(float value[8]) {
	memcpy(this->sdf, value, sizeof(float)*8);
}

void OctreeNode::setType(SpaceType type) {
	uint8_t mask = (0x1 << 0) | (0x1 << 1);
	uint8_t value = (type  == SpaceType::Solid ? 0x1 : 0x0) | (type  == SpaceType::Empty ? 0x1 : 0x0) << 1;
	this->bits = (this->bits & (mask ^ 0xff)) | value;
}

uint8_t OctreeNode::getSimplification() const {
	return (this->bits >> 2) & 0x1Fu;
}

void OctreeNode::setSimplification(uint8_t value){
	uint8_t mask = 0x7Cu;
	this->bits = (this->bits & ~mask) | ((value & 0x1Fu) << 2);
}

bool OctreeNode::isChunk() const {
	return this->bits & (0x1 << 7);
}
void OctreeNode::setChunk(bool value) {
	uint8_t mask = (0x1 << 7);
	this->bits = (this->bits & ~mask) | (value ? mask : 0x0);
}

bool OctreeNode::isLeaf() const {
	return this->blockId == UINT_MAX;
}


SpaceType OctreeNode::getType() const {
	if(this->bits & (0x1 << 0)) {
		return SpaceType::Solid;
	} else if(this->bits & (0x1 << 1)) {
		return SpaceType::Empty;
	} else {
		return SpaceType::Surface;
	}
}

OctreeNode * OctreeNode::compress(OctreeAllocator &allocator, BoundingCube * cube, const BoundingCube &chunk) {
	int intersectingChildCount = 0;
	int intersectingIndex = -1;

	for (int i = 0; i < 8; ++i) {
		if (cube->getChild(i).intersects(chunk)) {
			intersectingChildCount++;
			intersectingIndex = i;
		}
	}

	if(intersectingChildCount == 1) {
		ChildBlock * block = this->getBlock(allocator);
		if(block != NULL) {
			OctreeNode * childNode = block->get(intersectingIndex, allocator);
			if(childNode != NULL) {
				BoundingCube c = cube->getChild(intersectingIndex);
				*cube = c;
				return childNode->compress(allocator, cube, chunk);
			}
		}
	}
	return this;
}


uint OctreeNode::exportSerialization(OctreeAllocator &allocator, std::vector<OctreeNodeCubeSerialized> * nodes, int * leafNodes, const BoundingCube &cube, const BoundingCube &chunk, uint level) {
	if( this->getType() != SpaceType::Surface || !chunk.intersects(cube)) {
		return 0; // Skip this node
	}
	uint index = nodes->size(); 

	OctreeNodeCubeSerialized n(this->sdf, cube, this->vertex, this->bits, level);
	nodes->push_back(n);
	if(isLeaf()) {
		++(*leafNodes);
	}

	ChildBlock * block = this->getBlock(allocator);
	if(block != NULL) {
		for(int i=0; i < 8; ++i) {
			OctreeNode * childNode = block->get(i, allocator);
			if(childNode != NULL) {
				BoundingCube c = cube.getChild(i);
			    (*nodes)[index].children[i] = childNode->exportSerialization(allocator, nodes, leafNodes, c, chunk, level + 1);
			}
		}
	}
	return index;
}

void OctreeNode::setBrush(int brushIndex) {
	vertex.brushIndex = brushIndex;
}
int OctreeNode::getBrush() const {
	return this->vertex.brushIndex;
}