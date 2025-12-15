#pragma once
#include "../space/OctreeChangeHandler.hpp"
#include "OctreeLayer.tpp"
#include "../space/OctreeNode.hpp"
#include "../space/InstanceData.hpp"
#include "../space/DebugInstanceData.hpp"

class SolidSpaceChangeHandler : public OctreeChangeHandler {
	OctreeLayer<InstanceData> * solidInfo;

	public:
	SolidSpaceChangeHandler(
		OctreeLayer<InstanceData> * solidInfo
	);

	void create(OctreeNode* nodeId) override;
	void update(OctreeNode* nodeId) override;
	void erase(OctreeNode* nodeId) override;
};