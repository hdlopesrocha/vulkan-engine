#pragma once
#include "../space/OctreeChangeHandler.hpp"
#include "OctreeLayer.tpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeNodeData.hpp"
#include "../space/InstanceData.hpp"
#include <functional>

class LiquidSpaceChangeHandler : public OctreeChangeHandler {
	OctreeLayer<InstanceData> * liquidInfo;


	// Callback for when a node is updated (for mesh loading)
	NodeDataCallback onNodeUpdated;
	NodeDataCallback onNodeErased;

	public:
	LiquidSpaceChangeHandler(
		OctreeLayer<InstanceData> * liquidInfo
	);

	void create(OctreeNodeData& data) override;
	void update(OctreeNodeData& data) override;
	void erase(OctreeNodeData& data) override;

	// Set callbacks for mesh loading integration
	void setOnNodeUpdated(NodeDataCallback callback) { onNodeUpdated = callback; }
	void setOnNodeErased(NodeDataCallback callback) { onNodeErased = callback; }
};
