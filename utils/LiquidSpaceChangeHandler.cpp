#include "LiquidSpaceChangeHandler.hpp"

// Default empty callback used to initialize reference members
static NodeDataCallback emptyNodeDataCallback;


LiquidSpaceChangeHandler::LiquidSpaceChangeHandler(
		const NodeDataCallback& onNodeUpdatedCallback_,
    	const NodeDataCallback& onNodeDeletedCallback_
	)
    : onNodeUpdatedCallback(onNodeUpdatedCallback_),
      onNodeDeletedCallback(onNodeDeletedCallback_) {
}

void LiquidSpaceChangeHandler::onNodeAdded(const OctreeNodeData& nodeData) const {
    onNodeUpdatedCallback(nodeData);
};

void LiquidSpaceChangeHandler::onNodeDeleted(const OctreeNodeData& nodeData) const {
    onNodeDeletedCallback(nodeData);
};

