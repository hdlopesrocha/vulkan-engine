#pragma once
#include "../space/OctreeChangeHandler.hpp"
#include "OctreeLayer.tpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeNodeData.hpp"
#include "../space/InstanceData.hpp"
#include "../space/DebugInstanceData.hpp"
#include <functional>

class SolidSpaceChangeHandler : public OctreeChangeHandler {
	OctreeLayer<InstanceData> * solidInfo;

	// Callback for when a node is updated (for mesh loading)
	NodeDataCallback &onNodeUpdated;
	NodeDataCallback &onNodeCreated;
	NodeDataCallback &onNodeErased;

	public:
	SolidSpaceChangeHandler(
		OctreeLayer<InstanceData> * solidInfo
	);

	void create(OctreeNodeData& data) override;
	void update(OctreeNodeData& data) override;
	void erase(OctreeNodeData& data) override;

	// Set callbacks for mesh loading integration
	void setOnNodeUpdated(const NodeDataCallback &callback) { onNodeUpdated = callback; }
	void setOnNodeCreated(const NodeDataCallback &callback) { onNodeCreated = callback; }
	void setOnNodeErased(const NodeDataCallback &callback) { onNodeErased = callback; }
};