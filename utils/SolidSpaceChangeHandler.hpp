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
	std::function<void(const OctreeNodeData&)> onNodeUpdated;
	std::function<void(const OctreeNodeData&)> onNodeCreated;
	std::function<void(const OctreeNodeData&)> onNodeErased;

	public:
	SolidSpaceChangeHandler(
		OctreeLayer<InstanceData> * solidInfo
	);

	void create(OctreeNodeData& data) override;
	void update(OctreeNodeData& data) override;
	void erase(OctreeNodeData& data) override;

	// Set callbacks for mesh loading integration
	void setOnNodeUpdated(NodeDataCallback callback) { onNodeUpdated = callback; }
	void setOnNodeCreated(NodeDataCallback callback) { onNodeCreated = callback; }
	void setOnNodeErased(NodeDataCallback callback) { onNodeErased = callback; }
};