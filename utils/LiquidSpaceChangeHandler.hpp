#pragma once
#include "../space/OctreeChangeHandler.hpp"
#include "OctreeLayer.tpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeNodeData.hpp"
#include "../space/InstanceData.hpp"
#include <functional>
#include "UniqueOctreeChangeHandler.hpp"

class LiquidSpaceChangeHandler : public OctreeChangeHandler {

	const NodeDataCallback& onNodeUpdatedCallback;
    const NodeDataCallback& onNodeDeletedCallback;

	public:
	LiquidSpaceChangeHandler(
		const NodeDataCallback& onNodeUpdatedCallback,
    	const NodeDataCallback& onNodeDeletedCallback
	);

	void onNodeAdded(const OctreeNodeData& data) const override;
	void onNodeDeleted(const OctreeNodeData& data) const override;

};