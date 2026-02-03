#pragma once
#include "../space/OctreeChangeHandler.hpp"
#include "OctreeLayer.tpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeNodeData.hpp"
#include "../space/InstanceData.hpp"
#include <functional>

class LiquidSpaceChangeHandler : public OctreeChangeHandler {


	// Callback for when a node is updated (for mesh loading)
	const NodeDataCallback &onNodeUpdated;
	const NodeDataCallback &onNodeErased;

	public:
	LiquidSpaceChangeHandler(
		const NodeDataCallback & onNodeUpdated,
		const NodeDataCallback & onNodeErased
	);

	void onNodeAdded(const OctreeNodeData& data) const override;
	void onNodeDeleted(const OctreeNodeData& data) const override;

};