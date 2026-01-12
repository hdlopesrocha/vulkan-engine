#pragma once
#include "../space/OctreeChangeHandler.hpp"
#include "OctreeLayer.tpp"
#include "../space/OctreeNode.hpp"
#include "../space/InstanceData.hpp"
#include <functional>

class LiquidSpaceChangeHandler : public OctreeChangeHandler {
	OctreeLayer<InstanceData> * liquidInfo;

	// Callback for when a node is updated (for mesh loading)
	std::function<void(OctreeNode*)> onNodeUpdated;
	std::function<void(OctreeNode*)> onNodeErased;

	public:
	LiquidSpaceChangeHandler(
		OctreeLayer<InstanceData> * liquidInfo
	);

	void create(OctreeNode* nodeId) override;
	void update(OctreeNode* nodeId) override;
	void erase(OctreeNode* nodeId) override;

	// Set callbacks for mesh loading integration
	void setOnNodeUpdated(std::function<void(OctreeNode*)> callback) { onNodeUpdated = callback; }
	void setOnNodeErased(std::function<void(OctreeNode*)> callback) { onNodeErased = callback; }
};

