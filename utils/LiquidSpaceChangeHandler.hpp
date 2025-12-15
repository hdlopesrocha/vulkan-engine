#pragma once
#include "../space/OctreeChangeHandler.hpp"
#include "OctreeLayer.tpp"
#include "../space/OctreeNode.hpp"
#include "../space/InstanceData.hpp"

class LiquidSpaceChangeHandler : public OctreeChangeHandler {
	OctreeLayer<InstanceData> * liquidInfo;

	public:
	LiquidSpaceChangeHandler(
		OctreeLayer<InstanceData> * liquidInfo
	);

	void create(OctreeNode* nodeId) override;
	void update(OctreeNode* nodeId) override;
	void erase(OctreeNode* nodeId) override;
};

