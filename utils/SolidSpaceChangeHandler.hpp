#pragma once
#include "../space/OctreeChangeHandler.hpp"
#include "OctreeLayer.tpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeNodeData.hpp"
#include "../space/InstanceData.hpp"
#include "../space/DebugInstanceData.hpp"
#include <functional>

class SolidSpaceChangeHandler : public OctreeChangeHandler {

	// Callback for when a node is updated (for mesh loading)
	const NodeDataCallback &onNodeUpdated;
	const NodeDataCallback &onNodeErased;

	public:
	SolidSpaceChangeHandler(
		const NodeDataCallback & onNodeUpdated,
		const NodeDataCallback & onNodeErased
	);

	void onNodeAdded(const OctreeNodeData& data) const override;
	void onNodeDeleted(const OctreeNodeData& data) const override;

};