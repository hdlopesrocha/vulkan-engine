#pragma once
#include "../space/OctreeChangeHandler.hpp"
#include "OctreeLayer.tpp"
#include "../space/OctreeNode.hpp"
#include "../space/InstanceData.hpp"
#include "../space/DebugInstanceData.hpp"
#include <functional>

class SolidSpaceChangeHandler : public OctreeChangeHandler {
	OctreeLayer<InstanceData> * solidInfo;

	// Callback for when a node is updated (for mesh loading)
	std::function<void(OctreeNode*)> onNodeUpdated;
	std::function<void(OctreeNode*)> onNodeErased;

	public:
	SolidSpaceChangeHandler(
		OctreeLayer<InstanceData> * solidInfo
	);

	void create(OctreeNode* nodeId) override;
	void update(OctreeNode* nodeId) override;
	void erase(OctreeNode* nodeId) override;

	// Set callbacks for mesh loading integration
	void setOnNodeUpdated(std::function<void(OctreeNode*)> callback) { onNodeUpdated = callback; }
	void setOnNodeErased(std::function<void(OctreeNode*)> callback) { onNodeErased = callback; }
};