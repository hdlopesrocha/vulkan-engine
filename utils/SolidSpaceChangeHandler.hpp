#pragma once
#include "../space/OctreeChangeHandler.hpp"
#include "OctreeLayer.tpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeNodeData.hpp"
#include "../space/InstanceData.hpp"
#include "../space/DebugInstanceData.hpp"
#include <functional>
#include "UniqueOctreeChangeHandler.hpp"

class SolidSpaceChangeHandler : public OctreeChangeHandler {

	const NodeDataCallback& onNodeUpdatedCallback;
	const NodeDataCallback& onNodeDeletedCallback;
	public:
	SolidSpaceChangeHandler(
		const NodeDataCallback& onNodeUpdatedCallback,
    	const NodeDataCallback& onNodeDeletedCallback
	);

	void onNodeAdded(const OctreeNodeData& data) const override;
	void onNodeDeleted(const OctreeNodeData& data) const override;

};
